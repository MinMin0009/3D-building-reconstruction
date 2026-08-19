import os, json, random
import numpy as np
import cv2, torch, rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize
from sklearn.cluster import MeanShift

from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import pred_lines
from Tree_Segmentation.config import load_config

# ------------------------ Utils ------------------------

def _floor_angle_bin(a: float) -> int:
    return int(np.floor(a)) % 180

def angle_difference(a1, a2):
    a1 = a1 % 180.0; a2 = a2 % 180.0
    diff = abs(a1 - a2)
    return min(diff, 180.0 - diff)

def rotate_point(center, point, angle):
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x = point[0] - center[0]; y = point[1] - center[1]
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
    x1, y1 = map(int, pt1); x2, y2 = map(int, pt2)
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
        p1 = (int(l[0]), int(l[1])); p2 = (int(l[2]), int(l[3]))
        if not is_line_near_invalid_area(p1, p2, valid_mask_bool, threshold_pix, thickness):
            kept.append(l)
    return kept

# ------------------------ Main ------------------------
def main():
    # ======= Config =======
    config = load_config(None)  

    cropped_images_folder = os.path.join(config['paths']['result_path'], 'Building Candidate Images', 'building images')
    results_folder = os.path.join(config['paths']['result_path'], 'Lines detection')
    os.makedirs(results_folder, exist_ok=True)

    # ======= Model =======
    current_dir = os.path.dirname(__file__) or "./"
    model_path = os.path.join(current_dir, 'models', 'mlsd_large_512_fp32.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileV2_MLSD_Large().to(device).eval()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    # ======= Process TIFF =======
    for img_fn in os.listdir(cropped_images_folder):
        if not img_fn.lower().endswith('.tif'):
            continue

        img_path = os.path.join(cropped_images_folder, img_fn)

        with rasterio.open(img_path) as ds:

            img = ds.read([1, 2, 3])             # CxHxW
            img = np.transpose(img, (1, 2, 0))   # HxWxC
            geo_transform = ds.transform
            projection = ds.crs

            try:
                mask_u8 = ds.read_masks(1)  # HxW, uint8
                valid_mask = (mask_u8 > 0)
            except Exception:
                
                valid_mask = np.any(img != 0, axis=2)

        H, W = img.shape[:2]
        original_size = (H, W)

        # Resize 512x512 cho MLSD
        resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  

        resized_valid_mask = cv2.resize(
            valid_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Detect lines 
        with torch.no_grad():
            lines = pred_lines(resized_img, model, [512, 512], 0.1, 20)
        print(f"[INFO] {img_fn}: detected={len(lines)}")

        filtered_lines = filter_lines_near_invalid(lines, resized_valid_mask, threshold_pix=2, thickness=1)
        print(f"[INFO] {img_fn}: after_invalid_filter={len(filtered_lines)}")

        sx = W / 512.0; sy = H / 512.0
        scaled_filtered_lines = [(int(l[0]*sx), int(l[1]*sy), int(l[2]*sx), int(l[3]*sy))
                                 for l in filtered_lines]

        if len(scaled_filtered_lines) == 0:
            json_path = os.path.join(results_folder, f"{os.path.splitext(img_fn)[0]}_lines.json")
            with open(json_path, 'w') as geo_file:
                json.dump([], geo_file, indent=4)
            print(f"[WARN] {img_fn}: No lines after filter. Skip.")
            continue

        line_lengths, line_angles, line_points = [], [], []
        for l in scaled_filtered_lines:
            p1 = (int(l[0]), int(l[1])); p2 = (int(l[2]), int(l[3]))
            L = float(np.hypot(l[2]-l[0], l[3]-l[1]))
            ang = float(np.degrees(np.arctan2(l[3]-l[1], l[2]-l[0])) % 180.0)
            line_lengths.append(L)
            line_angles.append(ang)
            line_points.append((p1, p2))

        # Histogram
        angle_bins = np.arange(0, 180, 1, dtype=int)
        total_lengths = np.zeros(180, dtype=np.float64)
        for ang, L in zip(line_angles, line_lengths):
            total_lengths[_floor_angle_bin(ang)] += L

        # MeanShift
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

        # Pixel size
        pixel_size = compute_pixel_size_from_affine(geo_transform)

        threshold_in_pixels = (0.3 / pixel_size) if pixel_size > 0 else 0.3

        # Peak
        peak_angles = {}
        valid_clusters = []
        for lab in unique_labels:
            idxs = np.where(mean_shift_labels == lab)[0]
            if len(idxs) == 0: continue
            peak_idx = idxs[np.argmax(total_lengths[idxs])]
            peak_ang = int(angle_bins[peak_idx])  # integer bin
            if total_lengths[peak_idx] > threshold_in_pixels:
                peak_angles[lab] = peak_ang
                valid_clusters.append(lab)

        adjusted = dict(peak_angles)
        for i, l1 in enumerate(valid_clusters):
            for l2 in valid_clusters[i+1:]:
                p1, p2 = adjusted[l1], adjusted[l2]
                if 85.0 <= angle_difference(p1, p2) <= 95.0:
                    if total_lengths[p2] > total_lengths[p1]:
                        cand = _snap_to_supported_angle((p2 - 90) % 180, total_lengths, window_seq=(4, 6))
                        if cand is not None: adjusted[l1] = cand
                    elif total_lengths[p1] > total_lengths[p2]:
                        cand = _snap_to_supported_angle((p1 - 90) % 180, total_lengths, window_seq=(4, 6))
                        if cand is not None: adjusted[l2] = cand
        peak_angles = adjusted

        to_remove = set()
        for i, l1 in enumerate(valid_clusters):
            for l2 in valid_clusters[i+1:]:
                if angle_difference(peak_angles[l1], peak_angles[l2]) < 25.0:
                    t1 = total_lengths[_floor_angle_bin(peak_angles[l1])]
                    t2 = total_lengths[_floor_angle_bin(peak_angles[l2])]
                    if t1 < t2: to_remove.add(l1)
                    else:       to_remove.add(l2)
        valid_clusters = [lab for lab in valid_clusters if lab not in to_remove]

        for lab in valid_clusters:
            print(f"Cluster {lab+1}: Peak Angle = {peak_angles[lab]} degrees")

        filtered_lines2 = []
        for i, (p, L, ang) in enumerate(zip(line_points, line_lengths, line_angles)):
            bin_index = _floor_angle_bin(ang)
            lab = mean_shift_labels[bin_index]
            if lab in valid_clusters:
                filtered_lines2.append((p, L, lab))

        cluster_lines = {lab: [] for lab in valid_clusters}
        for line, L, lab in filtered_lines2:
            cluster_lines[lab].append((line, L))
        for lab in cluster_lines:
            cluster_lines[lab].sort(key=lambda x: x[1], reverse=True)

        distance_threshold_pixels = (0.5 / pixel_size) if pixel_size > 0 else 0.5

        shapely_lines = []
        refined_cluster_lines = {lab: [] for lab in cluster_lines}
        for lab, lines_to_consider in cluster_lines.items():
            removed = set()
            expanded = []
            for i, (line, L) in enumerate(lines_to_consider):
                pt1, pt2 = line
                mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
                cur = (np.degrees(np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])) % 180.0)
                tgt = peak_angles[lab]
                d = tgt - cur
                p1r = rotate_point(mid, pt1, d)
                p2r = rotate_point(mid, pt2, d)
                p1e, p2e = extend_line(p1r, p2r)
                expanded.append(((p1e, p2e), (pt1, pt2)))

            for i, (e1, org1) in enumerate(expanded):
                if i in removed: 
                    continue
                refined_cluster_lines[lab].append((org1, lines_to_consider[i][1]))
                shapely_lines.append(LineString([e1[0], e1[1]]))
                for j, (e2, org2) in enumerate(expanded[i+1:], start=i+1):
                    if j in removed: 
                        continue
                    if distance_between_parallel_lines(e1, e2) < distance_threshold_pixels:
                        removed.add(j)

        # Polygonize
        if len(shapely_lines):
            lines_union = unary_union(shapely_lines)
            polygons = list(polygonize(lines_union))
        else:
            polygons = []

        # JSON
        boxes_info = []
        for poly in polygons:
            if not poly.is_empty and isinstance(poly, Polygon):
                exterior = np.array(poly.exterior.coords, dtype=np.int32)
                boxes_info.append([{ "x": int(p[0]), "y": int(p[1]) } for p in exterior[:-1]])

        json_path = os.path.join(results_folder, f"{os.path.splitext(img_fn)[0]}_lines.json")
        with open(json_path, 'w') as f:
            json.dump(boxes_info, f, indent=4)

        color_img = img.copy()
        for poly in polygons:
            if not poly.is_empty and isinstance(poly, Polygon):
                exterior = np.array(poly.exterior.coords, dtype=np.int32)
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                cv2.fillPoly(color_img, [exterior], color)

        out_tif = os.path.join(results_folder, f"{os.path.splitext(img_fn)[0]}_lines.tif")
        with rasterio.open(out_tif, 'w',
                           driver='GTiff',
                           height=color_img.shape[0],
                           width=color_img.shape[1],
                           count=3,
                           dtype=color_img.dtype,
                           crs=projection,
                           transform=geo_transform) as dst:
            for i in range(3):
                dst.write(color_img[:, :, i], i+1)

        print(f"[DONE] {img_fn}: polygons={len(polygons)} saved.")

if __name__ == '__main__':
    main()
