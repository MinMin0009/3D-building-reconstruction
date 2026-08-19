import os
import json
import random
import re

import cv2
import laspy
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize
from sklearn.cluster import MeanShift

from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import deccode_output_score_and_ptss
from Tree_Segmentation.config import load_config


POINT_SUPPORT_MIN_RADIUS_PX = 5.0
POINT_SUPPORT_MAX_RADIUS_PX = 12.0
POINT_SUPPORT_RADIUS_SCALE_MNN = 1.75
POINT_SUPPORT_HARD_RADIUS_MULT = 2.0
POINT_SUPPORT_HARD_MAX_RADIUS_PX = 24.0
POINT_SUPPORT_SAMPLE_STEP_SCALE = 0.75
POINT_SUPPORT_WEIGHT_GAMMA = 2.0
POINT_SUPPORT_WEIGHT_FLOOR = 0.25


def _floor_angle_bin(a: float) -> int:
    return int(np.floor(a)) % 180


def angle_difference(a1, a2):
    a1 = a1 % 180.0
    a2 = a2 % 180.0
    diff = abs(a1 - a2)
    return min(diff, 180.0 - diff)


def rotate_point(center, point, angle):
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x = point[0] - center[0]
    y = point[1] - center[1]
    xr = x * cos_a - y * sin_a
    yr = x * sin_a + y * cos_a
    return (center[0] + xr, center[1] + yr)


def extend_line(pt1, pt2, length=20000):
    v = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]], dtype=np.float64)
    L = np.linalg.norm(v)
    if L == 0:
        return pt1, pt2
    u = v / L
    p1 = (pt1[0] - u[0] * length, pt1[1] - u[1] * length)
    p2 = (pt2[0] + u[0] * length, pt2[1] + u[1] * length)
    return p1, p2


def distance_between_parallel_lines(line1, line2):
    return LineString([line1[0], line1[1]]).distance(LineString([line2[0], line2[1]]))


def _assign_labels_by_centers(angle_bins, centers_1d):
    centers = np.array(centers_1d, dtype=np.float32).ravel() % 180.0
    a = angle_bins.astype(np.float32) % 180.0
    diffs = np.abs(a[:, None] - centers[None, :])
    cyc = np.minimum(diffs, 180.0 - diffs)
    return np.argmin(cyc, axis=1)


def _snap_to_supported_angle(target_angle, total_lengths, window_seq=(4, 6)):
    tgt = _floor_angle_bin(target_angle)
    for w in window_seq:
        cands = [(_floor_angle_bin(tgt + d)) for d in range(-w, w + 1)]
        cands = [c for c in cands if total_lengths[c] > 0]
        if cands:
            return max(cands, key=lambda c: total_lengths[c])
    return None


def compute_pixel_size_from_affine(T: Affine) -> float:
    px_x = float(np.hypot(T.a, T.b))
    px_y = float(np.hypot(T.d, T.e))
    if px_x > 0 and px_y > 0:
        return (px_x + px_y) / 2.0
    return max(px_x, px_y)


def is_line_near_invalid_area(pt1, pt2, valid_mask_bool, threshold_pix=2, thickness=1):
    x1, y1 = map(int, pt1)
    x2, y2 = map(int, pt2)
    h, w = valid_mask_bool.shape
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=thickness)
    ys, xs = np.where(line_mask == 255)
    if len(ys) == 0:
        return False
    invalid = (~valid_mask_bool[ys, xs]).sum()
    return invalid >= int(threshold_pix)


def filter_lines_near_invalid(lines, valid_mask_bool, threshold_pix=2, thickness=1):
    kept = []
    for l in lines:
        p1 = (int(l[0]), int(l[1]))
        p2 = (int(l[2]), int(l[3]))
        if not is_line_near_invalid_area(p1, p2, valid_mask_bool, threshold_pix, thickness):
            kept.append(l)
    return kept


def pred_lines_device_aware(image, model, input_shape=(512, 512), score_thr=0.10, dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate(
        [
            cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
            np.ones([input_shape[0], input_shape[1], 1]),
        ],
        axis=-1,
    )

    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype("float32")
    batch_image = (batch_image / 127.5) - 1.0

    device = next(model.parameters()).device
    batch_image = torch.from_numpy(batch_image).float().to(device)
    outputs = model(batch_image)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    if not segments_list:
        return np.empty((0, 4), dtype=np.float32)

    lines = 2 * np.array(segments_list, dtype=np.float32)
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio
    return lines


def find_matching_las(base_name, las_folder):
    m = re.search(r"(\d+)$", base_name)
    candidates = []
    if m:
        idx = m.group(1)
        candidates.append(os.path.join(las_folder, f"{idx}_point_cloud.las"))
    candidates.append(os.path.join(las_folder, f"{base_name}_point_cloud.las"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_las_points_in_pixel_space(las_path, geo_transform, width, height):
    las = laspy.read(las_path)
    xy = np.column_stack([las.x, las.y])
    cols, rows = (~geo_transform) * (xy[:, 0], xy[:, 1])
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not np.any(inside):
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([cols[inside], rows[inside]]).astype(np.float32)


def line_key(line, decimals=3):
    return tuple(round(float(v), decimals) for v in line)


def prepare_debug_image(img):
    out = img.copy()
    if out.dtype != np.uint8:
        vmin = float(np.percentile(out, 2))
        vmax = float(np.percentile(out, 98))
        denom = max(vmax - vmin, 1e-6)
        out = np.clip((out - vmin) * (255.0 / denom), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def draw_lines(image, lines, color, thickness=2):
    for line in lines:
        x1, y1, x2, y2 = line
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.line(image, pt1, pt2, color, thickness=thickness)


def save_mlsd_check_outputs(
    check_folder,
    base_name,
    img,
    detected_lines,
    invalid_kept,
    point_annotated,
    rotated_lines,
    final_lines,
):
    os.makedirs(check_folder, exist_ok=True)

    detected_img = prepare_debug_image(img)
    draw_lines(detected_img, detected_lines, (0, 255, 255), thickness=2)
    cv2.imwrite(os.path.join(check_folder, f"{base_name}_01_detected.png"), cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR))

    invalid_img = prepare_debug_image(img)
    invalid_kept_keys = {line_key(line) for line in invalid_kept}
    invalid_removed = [line for line in detected_lines if line_key(line) not in invalid_kept_keys]
    draw_lines(invalid_img, invalid_removed, (255, 0, 0), thickness=2)
    draw_lines(invalid_img, invalid_kept, (0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(check_folder, f"{base_name}_02_invalid_filter.png"), cv2.cvtColor(invalid_img, cv2.COLOR_RGB2BGR))

    support_img = prepare_debug_image(img)
    if point_annotated:
        point_kept = [item["line"] for item in point_annotated if item["hard_hit_count"] > 0]
        point_removed = [item["line"] for item in point_annotated if item["hard_hit_count"] == 0]
        draw_lines(support_img, point_removed, (255, 0, 0), thickness=2)
        draw_lines(support_img, point_kept, (0, 255, 0), thickness=2)
    else:
        draw_lines(support_img, invalid_kept, (0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(check_folder, f"{base_name}_03_point_support.png"), cv2.cvtColor(support_img, cv2.COLOR_RGB2BGR))

    rotated_img = prepare_debug_image(img)
    draw_lines(rotated_img, rotated_lines, (255, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(check_folder, f"{base_name}_04_rotated_lines.png"), cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR))

    final_img = prepare_debug_image(img)
    draw_lines(final_img, final_lines, (255, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(check_folder, f"{base_name}_05_final_lines.png"), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def estimate_median_nn_distance(points_pix, cap=2000):
    if points_pix.shape[0] < 2:
        return 0.0
    if points_pix.shape[0] > cap:
        idx = np.random.choice(points_pix.shape[0], cap, replace=False)
        pts = points_pix[idx]
    else:
        pts = points_pix
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.median(d[:, 1]))


def estimate_point_support_params(points_pix):
    mnn = estimate_median_nn_distance(points_pix)
    support_radius_pix = POINT_SUPPORT_MIN_RADIUS_PX
    if mnn > 0:
        support_radius_pix = max(support_radius_pix, POINT_SUPPORT_RADIUS_SCALE_MNN * mnn)
    support_radius_pix = min(support_radius_pix, POINT_SUPPORT_MAX_RADIUS_PX)
    hard_radius_pix = min(POINT_SUPPORT_HARD_MAX_RADIUS_PX, max(support_radius_pix, support_radius_pix * POINT_SUPPORT_HARD_RADIUS_MULT))
    sample_step_pix = max(2.0, support_radius_pix * POINT_SUPPORT_SAMPLE_STEP_SCALE)
    return float(support_radius_pix), float(hard_radius_pix), float(sample_step_pix), float(mnn)


def measure_line_point_support(line, point_tree, soft_radius_pix, hard_radius_pix, sample_step_pix):
    x1, y1, x2, y2 = map(float, line)
    line_length = float(np.hypot(x2 - x1, y2 - y1))
    n_samples = max(2, int(np.ceil(line_length / sample_step_pix)) + 1)

    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    samples = np.column_stack(
        [
            x1 + (x2 - x1) * ts,
            y1 + (y2 - y1) * ts,
        ]
    )
    dist, _ = point_tree.query(samples, k=1)
    soft_hits = dist <= soft_radius_pix
    hard_hits = dist <= hard_radius_pix
    support_ratio = float(np.mean(soft_hits))
    hard_hit_count = int(np.sum(hard_hits))
    min_dist = float(np.min(dist)) if dist.size > 0 else float("inf")
    return support_ratio, hard_hit_count, min_dist


def annotate_lines_with_point_support(lines, points_pix, soft_radius_pix, hard_radius_pix, sample_step_pix):
    if len(lines) == 0 or points_pix.shape[0] == 0:
        return []

    point_tree = cKDTree(points_pix)
    annotated = []
    for line in lines:
        support_ratio, hard_hit_count, min_dist = measure_line_point_support(
            line,
            point_tree,
            soft_radius_pix=soft_radius_pix,
            hard_radius_pix=hard_radius_pix,
            sample_step_pix=sample_step_pix,
        )
        annotated.append(
            {
                "line": line,
                "support_ratio": support_ratio,
                "hard_hit_count": hard_hit_count,
                "min_dist": min_dist,
            }
        )
    return annotated


def filter_lines_without_point_support(annotated_lines):
    return [item for item in annotated_lines if item["hard_hit_count"] > 0]


def line_support_weight(support_ratio):
    return float((POINT_SUPPORT_WEIGHT_FLOOR + (1.0 - POINT_SUPPORT_WEIGHT_FLOOR) * support_ratio) ** POINT_SUPPORT_WEIGHT_GAMMA)


def main():
    config = load_config(None)

    cropped_images_folder = os.path.join(config["paths"]["result_path"], "Building Candidate Images", "building images")
    results_folder = os.path.join(config["paths"]["result_path"], "Lines detection")
    check_folder = os.path.join(results_folder, "mlsd check")
    las_folder = os.path.join(config["paths"]["result_path"], "LAS")
    os.makedirs(results_folder, exist_ok=True)

    current_dir = os.path.dirname(__file__) or "./"
    model_path = os.path.join(current_dir, "models", "mlsd_large_512_fp32.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileV2_MLSD_Large().to(device).eval()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    for img_fn in os.listdir(cropped_images_folder):
        if not img_fn.lower().endswith(".tif"):
            continue

        img_path = os.path.join(cropped_images_folder, img_fn)
        base_name = os.path.splitext(img_fn)[0]

        with rasterio.open(img_path) as ds:
            img = ds.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0))
            geo_transform = ds.transform
            projection = ds.crs

            try:
                mask_u8 = ds.read_masks(1)
                valid_mask = mask_u8 > 0
            except Exception:
                valid_mask = np.any(img != 0, axis=2)

        H, W = img.shape[:2]

        resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        resized_valid_mask = cv2.resize(
            valid_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        with torch.no_grad():
            lines = pred_lines_device_aware(resized_img, model, (512, 512), 0.1, 20)
        print(f"[INFO] {img_fn}: detected={len(lines)}")

        scaled_detected_lines = [
            (float(l[0] * (W / 512.0)), float(l[1] * (H / 512.0)), float(l[2] * (W / 512.0)), float(l[3] * (H / 512.0)))
            for l in lines
        ]

        filtered_lines = filter_lines_near_invalid(lines, resized_valid_mask, threshold_pix=2, thickness=1)
        print(f"[INFO] {img_fn}: after_invalid_filter={len(filtered_lines)}")

        sx = W / 512.0
        sy = H / 512.0
        scaled_filtered_lines = [
            (float(l[0] * sx), float(l[1] * sy), float(l[2] * sx), float(l[3] * sy))
            for l in filtered_lines
        ]
        scaled_invalid_kept_lines = list(scaled_filtered_lines)

        las_path = find_matching_las(base_name, las_folder)
        line_support_lookup = {}
        annotated_lines = []
        if las_path is not None:
            points_pix = load_las_points_in_pixel_space(las_path, geo_transform, W, H)
            print(f"[INFO] {img_fn}: las_points_in_crop={len(points_pix)}")
            if len(points_pix) == 0:
                print(f"[WARN] {img_fn}: no LAS points fall inside crop. Skip point-support filter.")
            else:
                soft_radius_pix, hard_radius_pix, sample_step_pix, median_nn = estimate_point_support_params(points_pix)
                print(
                    f"[INFO] {img_fn}: point_support_radius_soft={soft_radius_pix:.2f}px "
                    f"hard={hard_radius_pix:.2f}px sample_step={sample_step_pix:.2f}px "
                    f"median_nn={median_nn:.2f}px"
                )
                annotated_lines = annotate_lines_with_point_support(
                    scaled_filtered_lines,
                    points_pix,
                    soft_radius_pix=soft_radius_pix,
                    hard_radius_pix=hard_radius_pix,
                    sample_step_pix=sample_step_pix,
                )
                scaled_filtered_lines = [item["line"] for item in filter_lines_without_point_support(annotated_lines)]
                line_support_lookup = {
                    tuple(item["line"]): item["support_ratio"]
                    for item in annotated_lines
                    if item["hard_hit_count"] > 0
                }
                print(f"[INFO] {img_fn}: after_point_support_filter={len(scaled_filtered_lines)}")
        else:
            print(f"[WARN] {img_fn}: matching LAS not found. Skip point-support filter.")

        if len(scaled_filtered_lines) == 0:
            save_mlsd_check_outputs(
                check_folder,
                base_name,
                img,
                scaled_detected_lines,
                scaled_invalid_kept_lines,
                annotated_lines,
                [],
                [],
            )
            json_path = os.path.join(results_folder, f"{base_name}_lines.json")
            with open(json_path, "w") as geo_file:
                json.dump([], geo_file, indent=4)
            print(f"[WARN] {img_fn}: No lines after filters. Skip.")
            continue

        line_lengths = []
        line_angles = []
        line_points = []
        for l in scaled_filtered_lines:
            p1 = (float(l[0]), float(l[1]))
            p2 = (float(l[2]), float(l[3]))
            L = float(np.hypot(l[2] - l[0], l[3] - l[1]))
            ang = float(np.degrees(np.arctan2(l[3] - l[1], l[2] - l[0])) % 180.0)
            line_lengths.append(L)
            line_angles.append(ang)
            line_points.append((p1, p2))

        angle_bins = np.arange(0, 180, 1, dtype=int)
        total_lengths = np.zeros(180, dtype=np.float64)
        for line, ang, L in zip(scaled_filtered_lines, line_angles, line_lengths):
            support_ratio = line_support_lookup.get(tuple(line), 1.0)
            total_lengths[_floor_angle_bin(ang)] += L * line_support_weight(support_ratio)

        X = np.array([[a] for a in angle_bins], dtype=np.float32)
        y = total_lengths
        weighted_X = np.repeat(X, y.astype(int), axis=0)
        if weighted_X.shape[0] == 0:
            nz = y > 0
            weighted_X = X[nz] if np.any(nz) else X

        ms = MeanShift(bandwidth=5, bin_seeding=True)
        ms.fit(weighted_X)
        try:
            mean_shift_labels = ms.predict(X)
        except AttributeError:
            centers = ms.cluster_centers_.ravel()
            mean_shift_labels = _assign_labels_by_centers(angle_bins, centers)

        unique_labels = np.unique(mean_shift_labels)

        pixel_size = compute_pixel_size_from_affine(geo_transform)
        threshold_in_pixels = (0.3 / pixel_size) if pixel_size > 0 else 0.3

        peak_angles = {}
        valid_clusters = []
        for lab in unique_labels:
            idxs = np.where(mean_shift_labels == lab)[0]
            if len(idxs) == 0:
                continue
            peak_idx = idxs[np.argmax(total_lengths[idxs])]
            peak_ang = int(angle_bins[peak_idx])
            if total_lengths[peak_idx] > threshold_in_pixels:
                peak_angles[lab] = peak_ang
                valid_clusters.append(lab)

        adjusted = dict(peak_angles)
        for i, l1 in enumerate(valid_clusters):
            for l2 in valid_clusters[i + 1 :]:
                p1, p2 = adjusted[l1], adjusted[l2]
                if 85.0 <= angle_difference(p1, p2) <= 95.0:
                    if total_lengths[p2] > total_lengths[p1]:
                        cand = _snap_to_supported_angle((p2 - 90) % 180, total_lengths, window_seq=(4, 6))
                        if cand is not None:
                            adjusted[l1] = cand
                    elif total_lengths[p1] > total_lengths[p2]:
                        cand = _snap_to_supported_angle((p1 - 90) % 180, total_lengths, window_seq=(4, 6))
                        if cand is not None:
                            adjusted[l2] = cand
        peak_angles = adjusted

        to_remove = set()
        for i, l1 in enumerate(valid_clusters):
            for l2 in valid_clusters[i + 1 :]:
                if angle_difference(peak_angles[l1], peak_angles[l2]) < 25.0:
                    t1 = total_lengths[_floor_angle_bin(peak_angles[l1])]
                    t2 = total_lengths[_floor_angle_bin(peak_angles[l2])]
                    if t1 < t2:
                        to_remove.add(l1)
                    else:
                        to_remove.add(l2)
        valid_clusters = [lab for lab in valid_clusters if lab not in to_remove]

        for lab in valid_clusters:
            print(f"Cluster {lab + 1}: Peak Angle = {peak_angles[lab]} degrees")

        filtered_lines2 = []
        for line, p, L, ang in zip(scaled_filtered_lines, line_points, line_lengths, line_angles):
            bin_index = _floor_angle_bin(ang)
            lab = mean_shift_labels[bin_index]
            if lab in valid_clusters:
                support_ratio = line_support_lookup.get(tuple(line), 1.0)
                filtered_lines2.append((p, L, lab, support_ratio))

        cluster_lines = {lab: [] for lab in valid_clusters}
        for line, L, lab, support_ratio in filtered_lines2:
            cluster_lines[lab].append((line, L, support_ratio))
        for lab in cluster_lines:
            cluster_lines[lab].sort(key=lambda x: x[1] * line_support_weight(x[2]), reverse=True)

        distance_threshold_pixels = (0.5 / pixel_size) if pixel_size > 0 else 0.5

        shapely_lines = []
        refined_cluster_lines = {lab: [] for lab in cluster_lines}
        rotated_lines_for_debug = []
        final_lines_for_debug = []
        for lab, lines_to_consider in cluster_lines.items():
            removed = set()
            expanded = []
            for i, (line, L, support_ratio) in enumerate(lines_to_consider):
                pt1, pt2 = line
                mid = ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0)
                cur = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])) % 180.0
                tgt = peak_angles[lab]
                d = tgt - cur
                p1r = rotate_point(mid, pt1, d)
                p2r = rotate_point(mid, pt2, d)
                rotated_lines_for_debug.append((p1r[0], p1r[1], p2r[0], p2r[1]))
                p1e, p2e = extend_line(p1r, p2r)
                expanded.append(((p1e, p2e), (p1r, p2r), (pt1, pt2), support_ratio))

            for i, (e1, rotated_seg_1, org1, support_ratio_1) in enumerate(expanded):
                if i in removed:
                    continue
                refined_cluster_lines[lab].append((org1, lines_to_consider[i][1], support_ratio_1))
                final_lines_for_debug.append(
                    (
                        rotated_seg_1[0][0],
                        rotated_seg_1[0][1],
                        rotated_seg_1[1][0],
                        rotated_seg_1[1][1],
                    )
                )
                shapely_lines.append(LineString([e1[0], e1[1]]))
                for j, (e2, rotated_seg_2, org2, support_ratio_2) in enumerate(expanded[i + 1 :], start=i + 1):
                    if j in removed:
                        continue
                    if distance_between_parallel_lines(e1, e2) < distance_threshold_pixels:
                        removed.add(j)

        if len(shapely_lines):
            lines_union = unary_union(shapely_lines)
            polygons = list(polygonize(lines_union))
        else:
            polygons = []

        boxes_info = []
        for poly in polygons:
            if not poly.is_empty and isinstance(poly, Polygon):
                exterior = np.array(poly.exterior.coords, dtype=np.int32)
                boxes_info.append([{"x": int(p[0]), "y": int(p[1])} for p in exterior[:-1]])

        save_mlsd_check_outputs(
            check_folder,
            base_name,
            img,
            scaled_detected_lines,
            scaled_invalid_kept_lines,
            annotated_lines,
            rotated_lines_for_debug,
            final_lines_for_debug,
        )

        json_path = os.path.join(results_folder, f"{base_name}_lines.json")
        with open(json_path, "w") as f:
            json.dump(boxes_info, f, indent=4)

        color_img = img.copy()
        for poly in polygons:
            if not poly.is_empty and isinstance(poly, Polygon):
                exterior = np.array(poly.exterior.coords, dtype=np.int32)
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                cv2.fillPoly(color_img, [exterior], color)

        out_tif = os.path.join(results_folder, f"{base_name}_lines.tif")
        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=color_img.shape[0],
            width=color_img.shape[1],
            count=3,
            dtype=color_img.dtype,
            crs=projection,
            transform=geo_transform,
        ) as dst:
            for i in range(3):
                dst.write(color_img[:, :, i], i + 1)

        print(f"[DONE] {img_fn}: polygons={len(polygons)} saved.")


if __name__ == "__main__":
    main()
