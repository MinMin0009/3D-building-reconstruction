import json
import os
import re

import alphashape
import laspy
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from collections import defaultdict
from matplotlib.path import Path
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

from Tree_Segmentation.config import load_config


config_path = None
config = load_config(config_path)

tif_folder = os.path.join(config["paths"]["result_path"], "Building Candidate Images", "building images")
json_folder = os.path.join(config["paths"]["result_path"], "Lines detection")
las_folder = os.path.join(config["paths"]["result_path"], "LAS")

result_path = config["paths"]["result_path"]
output_folder = os.path.join(result_path, "building box")
os.makedirs(output_folder, exist_ok=True)

debug_root = os.path.join(output_folder, "coverage debug")
os.makedirs(debug_root, exist_ok=True)


GRID_CELL_PX = 150
MIN_POINTS_IN_BOX = 3
EDGE_TOL = 0.25

COV_CELL_PX_BASE = 12
MAX_LOCAL_CELLS = 16000
DILATE_R_MIN = 0.75
DILATE_R_SCALE_MNN = 0.60
DILATE_R_SCALE_SIG = 1.25
MAX_DILATE_CELLS = 6

REFINE_BAND = 0.10
ALPHA_SAMPLE_CAP = 2000
UNIQUE_ROUND_DECIMALS = 2
K_PCA = 3.0
MIN_AREA_RATIO_FOR_ALPHA = 0.05

COVERAGE_THRESH = 0.95

PREVIEW_MAX_DIM = 2500
PREVIEW_MAX_POINTS = 12000
PREVIEW_POINT_R = 2

DEBUG_MAX_IMAGES_PER_SCENE = 40
DEBUG_DILATION_GAIN_THRESH = 0.20
DEBUG_LOW_RAW_THRESH = 0.60
DEBUG_EDGE_RATIO_THRESH = 0.70
DEBUG_LOCAL_TARGET_PX = 360
DEBUG_GRID_SCALE = 18

np.random.seed(0)


def _safe_area(geom):
    if geom is None:
        return 0.0
    if isinstance(geom, Polygon):
        return geom.area
    if isinstance(geom, MultiPolygon):
        return sum(g.area for g in geom.geoms)
    if isinstance(geom, GeometryCollection):
        return sum(_safe_area(g) for g in geom.geoms)
    return 0.0


def _dedup_points(pts, decimals=2):
    if len(pts) == 0:
        return pts
    pr = np.round(pts, decimals=decimals)
    _, idx = np.unique(pr, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _median_nn_dist(pts, cap=1500):
    if len(pts) < 3:
        return 0.0
    if len(pts) > cap:
        pts = pts[np.random.choice(len(pts), cap, replace=False)]
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.median(d[:, 1]))


def _pca_minor_sigma(pts):
    if len(pts) < 3:
        return 0.0
    X = pts - pts.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    n = max(1, X.shape[0] - 1)
    vars_ = (S ** 2) / n
    return float(np.sqrt(vars_[-1]))


def build_grid(points, cell=64):
    gx = np.floor(points[:, 0] / cell).astype(np.int32)
    gy = np.floor(points[:, 1] / cell).astype(np.int32)
    grid = {}
    for i, key in enumerate(zip(gx, gy)):
        grid.setdefault(key, []).append(i)
    return grid, cell


def query_grid(grid, cell, bbox):
    minx, miny, maxx, maxy = bbox
    gx0 = int(np.floor(minx / cell))
    gy0 = int(np.floor(miny / cell))
    gx1 = int(np.floor(maxx / cell))
    gy1 = int(np.floor(maxy / cell))
    out = []
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            out.extend(grid.get((gx, gy), ()))
    return np.array(out, dtype=np.int32) if out else np.empty(0, dtype=np.int32)


def _disk_struct(rcell):
    if rcell <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-rcell : rcell + 1, -rcell : rcell + 1]
    return (xx * xx + yy * yy) <= (rcell * rcell)


def _edge_point_ratio(poly, pts_in, edge_band_px):
    if len(pts_in) == 0:
        return 0.0
    pts = pts_in if len(pts_in) <= 1200 else pts_in[np.random.choice(len(pts_in), 1200, replace=False)]
    boundary = poly.boundary
    edge_hits = 0
    for x, y in pts:
        if boundary.distance(Point(float(x), float(y))) <= edge_band_px:
            edge_hits += 1
    return float(edge_hits) / float(len(pts))


def coverage_fast_grid_debug(box, points_pix, grid, cell):
    poly_xy = np.array([(p["x"], p["y"]) for p in box], dtype=float)
    poly = Polygon(poly_xy)
    debug = {
        "poly_xy": poly_xy,
        "bbox": None,
        "cov_raw": 0.0,
        "cov_dilated": 0.0,
        "cov_alpha": 0.0,
        "cov_final": 0.0,
        "denom": 0,
        "occ_raw_cells": 0,
        "occ_dilated_cells": 0,
        "n_in": 0,
        "mnn": 0.0,
        "sig": 0.0,
        "cell_px": float(COV_CELL_PX_BASE),
        "r_px": 0.0,
        "rcell": 0,
        "nx": 0,
        "ny": 0,
        "edge_point_ratio": 0.0,
        "used_refine": False,
        "alpha_param": None,
        "pts_in": np.empty((0, 2), dtype=float),
        "inside_cell": None,
        "occ": None,
        "occ_dil": None,
    }

    if (not poly.is_valid) or poly.area <= 0:
        return 0.0, 0, 0.0, 0.0, debug

    minx, miny = poly_xy.min(axis=0)
    maxx, maxy = poly_xy.max(axis=0)
    debug["bbox"] = [float(minx), float(miny), float(maxx), float(maxy)]

    cand_idx = query_grid(grid, cell, (minx, miny, maxx, maxy))
    if cand_idx.size == 0:
        return 0.0, 0, 0.0, 0.0, debug
    cand = points_pix[cand_idx]

    path = Path(poly_xy)
    in_mask = path.contains_points(cand, radius=EDGE_TOL)
    pts = cand[in_mask]
    n_in = len(pts)
    debug["n_in"] = int(n_in)
    debug["pts_in"] = pts
    if n_in < MIN_POINTS_IN_BOX:
        return 0.0, n_in, 0.0, 0.0, debug

    pts_use = pts if n_in <= 1500 else pts[np.random.choice(n_in, 1500, replace=False)]
    pts_use = _dedup_points(pts_use, decimals=UNIQUE_ROUND_DECIMALS)
    mnn = _median_nn_dist(pts_use)
    sig = _pca_minor_sigma(pts_use)
    debug["mnn"] = float(mnn)
    debug["sig"] = float(sig)

    cell_px = float(COV_CELL_PX_BASE)
    w = maxx - minx
    h = maxy - miny
    nx = max(1, int(np.ceil(w / cell_px)))
    ny = max(1, int(np.ceil(h / cell_px)))
    if nx * ny > MAX_LOCAL_CELLS:
        s = np.sqrt((nx * ny) / MAX_LOCAL_CELLS)
        cell_px *= s
        nx = max(1, int(np.ceil(w / cell_px)))
        ny = max(1, int(np.ceil(h / cell_px)))

    ix = np.clip(((pts[:, 0] - minx) / cell_px).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - miny) / cell_px).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), dtype=bool)
    occ[iy, ix] = True

    cx = (np.arange(nx) + 0.5) * cell_px + minx
    cy = (np.arange(ny) + 0.5) * cell_px + miny
    CX, CY = np.meshgrid(cx, cy)
    centers = np.column_stack([CX.ravel(), CY.ravel()])
    inside_cell = path.contains_points(centers, radius=EDGE_TOL).reshape(ny, nx)
    denom = int(inside_cell.sum())

    debug["cell_px"] = float(cell_px)
    debug["nx"] = int(nx)
    debug["ny"] = int(ny)
    debug["denom"] = int(denom)
    debug["inside_cell"] = inside_cell
    debug["occ"] = occ

    if denom == 0:
        return 0.0, n_in, mnn, sig, debug

    occ_raw_cells = int((occ & inside_cell).sum())
    cov_raw = float(occ_raw_cells) / float(denom)

    r_px = max(DILATE_R_MIN, DILATE_R_SCALE_MNN * mnn, DILATE_R_SCALE_SIG * sig)
    rcell = int(np.ceil(r_px / cell_px))
    rcell = int(min(rcell, MAX_DILATE_CELLS))
    struct = _disk_struct(rcell)
    occ_dil = binary_dilation(occ, structure=struct)
    occ_dilated_cells = int((occ_dil & inside_cell).sum())
    cov_grid = float(occ_dilated_cells) / float(denom)

    edge_band_px = max(2.0, 0.5 * cell_px)
    edge_ratio = _edge_point_ratio(poly, pts, edge_band_px)

    debug["cov_raw"] = float(cov_raw)
    debug["cov_dilated"] = float(cov_grid)
    debug["occ_raw_cells"] = int(occ_raw_cells)
    debug["occ_dilated_cells"] = int(occ_dilated_cells)
    debug["r_px"] = float(r_px)
    debug["rcell"] = int(rcell)
    debug["occ_dil"] = occ_dil
    debug["edge_point_ratio"] = float(edge_ratio)

    return cov_grid, n_in, mnn, sig, debug


def coverage_refine_alpha(box, pts_in, poly, area, mnn, sig):
    if len(pts_in) < 4 or mnn <= 0:
        return 0.0
    pts = pts_in if len(pts_in) <= ALPHA_SAMPLE_CAP else pts_in[np.random.choice(len(pts_in), ALPHA_SAMPLE_CAP, replace=False)]
    pts = _dedup_points(pts, decimals=UNIQUE_ROUND_DECIMALS)
    try:
        alpha = 1.0 / (2.0 * mnn)
        shape = alphashape.alphashape([tuple(p) for p in pts], alpha)
    except Exception:
        shape = None

    if (shape is None) or (_safe_area(shape) == 0.0) or ((_safe_area(shape) / area) < MIN_AREA_RATIO_FOR_ALPHA):
        hull = MultiPoint(pts).convex_hull
        shape = hull.buffer(max(0.75, max(K_PCA * sig, 0.5 * mnn)))

    return _safe_area(shape.intersection(poly)) / area


def coverage_ultrafast_debug(box, points_pix, grid, cell):
    poly_xy = np.array([(p["x"], p["y"]) for p in box], dtype=float)
    poly = Polygon(poly_xy)
    area = poly.area

    cov_grid, n_in, mnn, sig, debug = coverage_fast_grid_debug(box, points_pix, grid, cell)
    if n_in < MIN_POINTS_IN_BOX or area <= 0:
        debug["cov_final"] = 0.0
        return 0.0, n_in, None, False, sig, debug

    cov = cov_grid
    used_refine = False
    cov_alpha = 0.0
    if abs(cov_grid - COVERAGE_THRESH) < REFINE_BAND:
        minx, miny = poly_xy.min(axis=0)
        maxx, maxy = poly_xy.max(axis=0)
        cand_idx = query_grid(grid, cell, (minx, miny, maxx, maxy))
        cand = points_pix[cand_idx]
        path = Path(poly_xy)
        in_mask = path.contains_points(cand, radius=EDGE_TOL)
        pts_in = cand[in_mask]
        cov_alpha = coverage_refine_alpha(box, pts_in, poly, area, mnn, sig)
        cov = max(cov_grid, cov_alpha)
        used_refine = True

    debug["cov_alpha"] = float(cov_alpha)
    debug["cov_final"] = float(min(max(cov, 0.0), 1.0))
    debug["used_refine"] = bool(used_refine)
    debug["alpha_param"] = float(1.0 / (2.0 * mnn)) if mnn > 0 else None

    return float(min(max(cov, 0.0), 1.0)), n_in, (1.0 / (2.0 * mnn) if mnn > 0 else None), used_refine, float(sig), debug


def find_matching_las(base_name):
    m = re.search(r"(\d+)$", base_name)
    candidates = []
    if m:
        idx = m.group(1)
        candidates.append(os.path.join(las_folder, f"{idx}_point_cloud.las"))
    candidates.append(os.path.join(las_folder, f"{base_name}_point_cloud.las"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def save_preview_fast(tif_path, boxes_data, survivors_idx, survivors_f2_idx, pix, out_png):
    with rasterio.open(tif_path) as ds:
        W, H = ds.width, ds.height
        scale = min(1.0, PREVIEW_MAX_DIM / max(W, H))
        out_h = max(1, int(H * scale))
        out_w = max(1, int(W * scale))

        if ds.count >= 3:
            arr = ds.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
            arr = np.moveaxis(arr, 0, 2)
        else:
            a1 = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            arr = np.stack([a1, a1, a1], axis=2)

    if arr.dtype != np.uint8:
        vmin = float(np.percentile(arr, 2))
        vmax = float(np.percentile(arr, 98))
        denom = max(vmax - vmin, 1e-6)
        arr = np.clip((arr - vmin) * (255.0 / denom), 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    def _draw_box(coords, color, width):
        draw.polygon([(p["x"] * scale, p["y"] * scale) for p in coords], outline=color, width=width)

    for box in boxes_data:
        _draw_box(box, (255, 0, 0), 2)
    for idx in survivors_idx:
        _draw_box(boxes_data[idx], (0, 255, 0), 4)
    for idx in survivors_f2_idx:
        _draw_box(boxes_data[idx], (255, 215, 0), 4)
    if len(pix) > 0:
        pts = pix * scale
        step = max(1, len(pts) // PREVIEW_MAX_POINTS)
        r = max(1, int(PREVIEW_POINT_R * max(1.0, scale)))
        for x, y in pts[::step]:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 0, 255))

    img.save(out_png)


def _render_local_geometry_panel(box, debug):
    poly_xy = np.array([(p["x"], p["y"]) for p in box], dtype=float)
    minx, miny, maxx, maxy = debug["bbox"]
    pad = max(4.0, debug["cell_px"])
    origin_x = minx - pad
    origin_y = miny - pad
    width = max(1.0, maxx - minx + 2.0 * pad)
    height = max(1.0, maxy - miny + 2.0 * pad)
    scale = max(2, min(14, int(np.ceil(DEBUG_LOCAL_TARGET_PX / max(width, height)))))

    canvas_w = int(np.ceil(width * scale))
    canvas_h = int(np.ceil(height * scale))
    img = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(img)

    poly_pts = [((x - origin_x) * scale, (y - origin_y) * scale) for x, y in poly_xy]
    draw.polygon(poly_pts, outline=(255, 230, 0), width=3)

    pts_in = debug["pts_in"]
    r = max(1, min(4, scale // 3))
    for x, y in pts_in:
        cx = (float(x) - origin_x) * scale
        cy = (float(y) - origin_y) * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 170, 255))

    return img


def _render_grid_panel(debug):
    inside = debug["inside_cell"]
    occ = debug["occ"]
    occ_dil = debug["occ_dil"]
    if inside is None or occ is None or occ_dil is None:
        return Image.new("RGB", (200, 200), (255, 255, 255))

    rgb = np.zeros((inside.shape[0], inside.shape[1], 3), dtype=np.uint8)
    rgb[inside] = (65, 65, 65)
    rgb[(occ_dil & inside) & (~occ)] = (255, 195, 0)
    rgb[occ & inside] = (40, 170, 255)

    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((inside.shape[1] * DEBUG_GRID_SCALE, inside.shape[0] * DEBUG_GRID_SCALE), resample=Image.NEAREST)


def save_box_debug_panel(out_path, box_idx, box, metric, debug):
    geom_panel = _render_local_geometry_panel(box, debug)
    grid_panel = _render_grid_panel(debug)

    header_h = 96
    out_w = geom_panel.width + grid_panel.width
    out_h = header_h + max(geom_panel.height, grid_panel.height)
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    line1 = f"Box {box_idx} | kept_f1={metric['kept_f1']} rescued_f2={metric['rescued_f2']} suspicious={metric['debug_suspicious']}"
    line2 = (
        f"cov_raw={metric['cov_raw']:.3f} cov_dilated={metric['cov_dilated']:.3f} "
        f"cov_alpha={metric['cov_alpha']:.3f} cov_final={metric['cov_final']:.3f}"
    )
    line3 = (
        f"n_in={metric['n_in']} edge_ratio={metric['edge_point_ratio']:.3f} "
        f"mnn={metric['mnn']:.3f} sig={metric['sigma_minor']:.3f}"
    )
    line4 = (
        f"cell_px={metric['cell_px']:.2f} r_px={metric['r_px']:.2f} rcell={metric['rcell']} "
        f"denom={metric['denom']} occ_raw={metric['occ_raw_cells']} occ_dilated={metric['occ_dilated_cells']}"
    )
    line5 = "left: polygon + points in box | right: blue=occupied cells, yellow=added by dilation"

    draw.text((10, 8), line1, fill=(0, 0, 0))
    draw.text((10, 26), line2, fill=(0, 0, 0))
    draw.text((10, 44), line3, fill=(0, 0, 0))
    draw.text((10, 62), line4, fill=(0, 0, 0))
    draw.text((10, 80), line5, fill=(0, 0, 0))

    canvas.paste(geom_panel, (0, header_h))
    canvas.paste(grid_panel, (geom_panel.width, header_h))
    canvas.save(out_path)


def _box_debug_metric(box_idx, kept_f1, cov_final, n_in, a_used, refined, sig, debug):
    cov_raw = float(debug.get("cov_raw", 0.0))
    cov_dilated = float(debug.get("cov_dilated", 0.0))
    cov_alpha = float(debug.get("cov_alpha", 0.0))
    gain = cov_dilated - cov_raw
    edge_ratio = float(debug.get("edge_point_ratio", 0.0))
    suspicious = bool(
        kept_f1
        and (
            gain >= DEBUG_DILATION_GAIN_THRESH
            or (cov_raw < DEBUG_LOW_RAW_THRESH and edge_ratio >= DEBUG_EDGE_RATIO_THRESH)
            or (cov_raw < COVERAGE_THRESH and cov_final >= COVERAGE_THRESH)
        )
    )
    return {
        "index": int(box_idx),
        "kept_f1": bool(kept_f1),
        "rescued_f2": False,
        "cov_final": float(cov_final),
        "cov_raw": cov_raw,
        "cov_dilated": cov_dilated,
        "cov_alpha": cov_alpha,
        "gain_from_dilation": float(gain),
        "n_in": int(n_in),
        "alpha": None if a_used is None else float(a_used),
        "used_refine": bool(refined),
        "sigma_minor": float(sig),
        "mnn": float(debug.get("mnn", 0.0)),
        "cell_px": float(debug.get("cell_px", 0.0)),
        "r_px": float(debug.get("r_px", 0.0)),
        "rcell": int(debug.get("rcell", 0)),
        "denom": int(debug.get("denom", 0)),
        "occ_raw_cells": int(debug.get("occ_raw_cells", 0)),
        "occ_dilated_cells": int(debug.get("occ_dilated_cells", 0)),
        "edge_point_ratio": edge_ratio,
        "nx": int(debug.get("nx", 0)),
        "ny": int(debug.get("ny", 0)),
        "debug_suspicious": suspicious,
    }


def _suspicious_score(metric):
    return (
        metric["gain_from_dilation"] * 2.0
        + max(0.0, COVERAGE_THRESH - metric["cov_raw"])
        + 0.5 * metric["edge_point_ratio"]
    )


tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith(".tif")]
tif_files.sort()

for tif_file in tif_files:
    base_name = os.path.splitext(tif_file)[0]
    tif_path = os.path.join(tif_folder, tif_file)
    json_path = os.path.join(json_folder, f"{base_name}_lines.json")
    las_path = find_matching_las(base_name)

    if not os.path.exists(json_path) or las_path is None:
        print(f"[SKIP] {base_name}: missing JSON or LAS. json={os.path.exists(json_path)}, las={las_path is not None}")
        continue

    scene_debug_dir = os.path.join(debug_root, base_name)
    os.makedirs(scene_debug_dir, exist_ok=True)

    with rasterio.open(tif_path) as tif:
        T = tif.transform
        W, H = tif.width, tif.height
        crs = tif.crs

    with open(json_path, "r") as f:
        boxes_data = json.load(f)
    if not boxes_data:
        print(f"[WARN] {base_name}: boxes empty.")
        continue

    las = laspy.read(las_path)
    xy = np.column_stack([las.x, las.y])
    cols, rows = (~T) * (xy[:, 0], xy[:, 1])
    inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    pix = np.column_stack([cols[inside], rows[inside]])

    print(f"\n=== {base_name} ===")
    print(f"TIFF: {W}x{H}, CRS={crs}, LAS in-image: {inside.sum()} / {len(xy)} ({inside.mean():.2%}), boxes: {len(boxes_data)}")

    grid, cell = build_grid(pix, GRID_CELL_PX)
    polys = [Polygon([(p["x"], p["y"]) for p in box]) for box in boxes_data]
    valid_mask = [poly.is_valid and (poly.area > 0) for poly in polys]

    survivors = []
    survivors_idx = []
    kept = 0
    dropped_lowpts = 0
    dropped_cov = 0
    box_logs = []
    box_metrics = []
    suspicious_candidates = []

    for i, box in enumerate(boxes_data):
        if not valid_mask[i]:
            dropped_cov += 1
            metric = {
                "index": int(i),
                "kept_f1": False,
                "rescued_f2": False,
                "cov_final": 0.0,
                "cov_raw": 0.0,
                "cov_dilated": 0.0,
                "cov_alpha": 0.0,
                "gain_from_dilation": 0.0,
                "n_in": 0,
                "alpha": None,
                "used_refine": False,
                "sigma_minor": 0.0,
                "mnn": 0.0,
                "cell_px": 0.0,
                "r_px": 0.0,
                "rcell": 0,
                "denom": 0,
                "occ_raw_cells": 0,
                "occ_dilated_cells": 0,
                "edge_point_ratio": 0.0,
                "nx": 0,
                "ny": 0,
                "debug_suspicious": False,
            }
            box_metrics.append(metric)
            box_logs.append((i, 0, 0.0, 0.0, None, False, 0.0))
            continue

        cov, n_in, a_used, refined, sig, debug = coverage_ultrafast_debug(box, pix, grid, cell)
        area = float(polys[i].area)
        kept_f1 = bool(n_in >= MIN_POINTS_IN_BOX and cov >= COVERAGE_THRESH)

        if n_in < MIN_POINTS_IN_BOX:
            dropped_lowpts += 1
        elif kept_f1:
            survivors.append(box)
            survivors_idx.append(i)
            kept += 1
        else:
            dropped_cov += 1

        metric = _box_debug_metric(i, kept_f1, cov, n_in, a_used, refined, sig, debug)
        box_metrics.append(metric)
        box_logs.append((i, n_in, cov, area, a_used, refined, sig))

        if metric["debug_suspicious"]:
            suspicious_candidates.append((_suspicious_score(metric), i, box, debug))

    valid_idx = [i for i, vm in enumerate(valid_mask) if vm]
    valid_geoms = [polys[i] for i in valid_idx]
    survivors_f2_idx = []

    if len(valid_geoms) > 0:
        tree = STRtree(valid_geoms)
        wkb2orig = defaultdict(list)
        for i in valid_idx:
            wkb2orig[polys[i].wkb].append(i)

        kept_set = set(survivors_idx)
        dropped_idx = [i for i in range(len(boxes_data)) if valid_mask[i] and (i not in kept_set)]

        def _normalize_candidate_indices(cands):
            if cands is None:
                return []
            if isinstance(cands, np.ndarray) and np.issubdtype(cands.dtype, np.integer):
                return [valid_idx[int(k)] for k in cands.tolist()]
            out = []
            for g in cands:
                if g is not None:
                    out.extend(wkb2orig.get(g.wkb, []))
            return out

        def edge_neighbors(i):
            Pi = polys[i]
            try:
                cands = tree.query(Pi, predicate="touches")
            except TypeError:
                cands = tree.query(Pi)
            cand_idx = _normalize_candidate_indices(cands)
            nei = []
            for j in cand_idx:
                if j == i or not valid_mask[j]:
                    continue
                inter = Pi.intersection(polys[j])
                if inter.length > 1e-9 and inter.geom_type in ("LineString", "LinearRing", "MultiLineString"):
                    nei.append(j)
            return nei

        for i in dropped_idx:
            nei = edge_neighbors(i)
            if nei and all((j in kept_set) for j in nei):
                survivors_f2_idx.append(i)

    survivors_f2 = [boxes_data[i] for i in survivors_f2_idx]
    final_boxes = survivors + survivors_f2

    rescued_set = set(survivors_f2_idx)
    for metric in box_metrics:
        metric["rescued_f2"] = metric["index"] in rescued_set

    out_json = os.path.join(output_folder, f"{base_name}_final_boxes_data.json")
    with open(out_json, "w") as f:
        json.dump(final_boxes, f, indent=2)
    print(f"[SAVE] F1: {len(survivors)} | F2: {len(survivors_f2)} | drop_lowpts={dropped_lowpts}, drop_cov={dropped_cov} -> {out_json}")

    out_png = os.path.join(output_folder, f"{base_name}_boxes_visualization.png")
    save_preview_fast(tif_path, boxes_data, survivors_idx, survivors_f2_idx, pix, out_png)
    print(f"[SAVE] preview: {out_png}")

    debug_overview_png = os.path.join(scene_debug_dir, f"{base_name}_overview.png")
    save_preview_fast(tif_path, boxes_data, survivors_idx, survivors_f2_idx, pix, debug_overview_png)

    metrics_json = os.path.join(scene_debug_dir, f"{base_name}_coverage_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(box_metrics, f, indent=2)
    print(f"[DEBUG] metrics: {metrics_json}")

    suspicious_candidates.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, idx, box, debug) in enumerate(suspicious_candidates[:DEBUG_MAX_IMAGES_PER_SCENE], start=1):
        out_panel = os.path.join(scene_debug_dir, f"{rank:02d}_box_{idx:04d}.png")
        save_box_debug_panel(out_panel, idx, box, box_metrics[idx], debug)

    print(f"[DEBUG] suspicious box panels: {min(len(suspicious_candidates), DEBUG_MAX_IMAGES_PER_SCENE)} saved to {scene_debug_dir}")

    box_logs.sort(key=lambda t: t[1], reverse=True)
    print("Top boxes by points_in:")
    for idx, n_in, cov, area, a, refined, sig in box_logs[:10]:
        print(
            f"  Box {idx:4d}  pts_in={n_in:6d}  coverage={cov:6.3f}  area={area:.1f}  "
            f"alpha={('%.4f' % a) if a is not None else 'NA'}  refined={refined}  sigma_minor={sig:.3f}"
        )
