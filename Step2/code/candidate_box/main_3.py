import os, re, json, numpy as np, rasterio, laspy
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, MultiPoint
from shapely.ops import unary_union
from shapely.strtree import STRtree
from collections import defaultdict
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from rasterio.enums import Resampling
import alphashape

from Tree_Segmentation.config import load_config

# ===================== CONFIG & IO =====================
config_path = None
config = load_config(config_path)

tif_folder  = os.path.join(config['paths']['result_path'], 'Building Candidate Images', 'building images')
json_folder = os.path.join(config['paths']['result_path'], 'Lines detection')
las_folder  = os.path.join(config['paths']['result_path'], 'LAS')

result_path    = config['paths']['result_path']
output_folder  = os.path.join(result_path, "building box")
os.makedirs(output_folder, exist_ok=True)

# ===================== KNOBS =====================

GRID_CELL_PX           = 150     
MIN_POINTS_IN_BOX      = 3
EDGE_TOL               = 0.25    


COV_CELL_PX_BASE       = 12      
MAX_LOCAL_CELLS        = 16000   
DILATE_R_MIN           = 0.75    
DILATE_R_SCALE_MNN     = 0.60    # r >= 0.60 * medianNN
DILATE_R_SCALE_SIG     = 1.25    # r >= 1.25 * sigma_minor
MAX_DILATE_CELLS       = 6      


REFINE_BAND            = 0.10    
ALPHA_SAMPLE_CAP       = 2000
UNIQUE_ROUND_DECIMALS  = 2
K_PCA                  = 3.0
MIN_AREA_RATIO_FOR_ALPHA = 0.05

# Quyết định giữ
COVERAGE_THRESH        = 0.99


PREVIEW_MAX_DIM        = 2500   
PREVIEW_MAX_POINTS     = 12000
PREVIEW_POINT_R        = 2

np.random.seed(0)

# ===================== HELPERS =====================
def _safe_area(geom):
    if geom is None: return 0.0
    if isinstance(geom, Polygon): return geom.area
    if isinstance(geom, MultiPolygon): return sum(g.area for g in geom.geoms)
    if isinstance(geom, GeometryCollection): return sum(_safe_area(g) for g in geom.geoms)
    return 0.0

def _dedup_points(pts, decimals=2):
    if len(pts) == 0: return pts
    pr = np.round(pts, decimals=decimals)
    _, idx = np.unique(pr, axis=0, return_index=True)
    return pts[np.sort(idx)]

def _median_nn_dist(pts, cap=1500):
    if len(pts) < 3: return 0.0
    if len(pts) > cap:
        pts = pts[np.random.choice(len(pts), cap, replace=False)]
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.median(d[:, 1]))

def _pca_minor_sigma(pts):
    if len(pts) < 3: return 0.0
    X = pts - pts.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n = max(1, X.shape[0] - 1)
    vars_ = (S**2) / n
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
    gx0 = int(np.floor(minx / cell)); gy0 = int(np.floor(miny / cell))
    gx1 = int(np.floor(maxx / cell)); gy1 = int(np.floor(maxy / cell))
    out = []
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            out.extend(grid.get((gx, gy), ()))
    return np.array(out, dtype=np.int32) if out else np.empty(0, dtype=np.int32)

def _disk_struct(rcell):
    if rcell <= 0: return np.ones((1,1), dtype=bool)
    d = 2*rcell+1
    yy, xx = np.ogrid[-rcell:rcell+1, -rcell:rcell+1]
    return (xx*xx + yy*yy) <= (rcell*rcell)

def coverage_fast_grid(box, points_pix, grid, cell):

    poly_xy = np.array([(p['x'], p['y']) for p in box], dtype=float)
    poly = Polygon(poly_xy)
    if (not poly.is_valid) or poly.area <= 0:
        return 0.0, 0, 0.0, 0.0

    minx, miny = poly_xy.min(axis=0); maxx, maxy = poly_xy.max(axis=0)

    cand_idx = query_grid(grid, cell, (minx, miny, maxx, maxy))
    if cand_idx.size == 0:
        return 0.0, 0, 0.0, 0.0
    cand = points_pix[cand_idx]

    path = Path(poly_xy)
    in_mask = path.contains_points(cand, radius=EDGE_TOL)
    pts = cand[in_mask]
    n_in = len(pts)
    if n_in < MIN_POINTS_IN_BOX:
        return 0.0, n_in, 0.0, 0.0

    pts_use = pts if n_in <= 1500 else pts[np.random.choice(n_in, 1500, replace=False)]
    pts_use = _dedup_points(pts_use, decimals=UNIQUE_ROUND_DECIMALS)
    mnn  = _median_nn_dist(pts_use)
    sig  = _pca_minor_sigma(pts_use)

    cell_px = float(COV_CELL_PX_BASE)
    w = maxx - minx; h = maxy - miny
    nx = max(1, int(np.ceil(w / cell_px))); ny = max(1, int(np.ceil(h / cell_px)))
    if nx * ny > MAX_LOCAL_CELLS:
        s = np.sqrt((nx * ny) / MAX_LOCAL_CELLS)
        cell_px *= s
        nx = max(1, int(np.ceil(w / cell_px))); ny = max(1, int(np.ceil(h / cell_px)))

    ix = np.clip(((pts[:,0] - minx) / cell_px).astype(int), 0, nx-1)
    iy = np.clip(((pts[:,1] - miny) / cell_px).astype(int), 0, ny-1)
    occ = np.zeros((ny, nx), dtype=bool)
    occ[iy, ix] = True

    cx = (np.arange(nx) + 0.5) * cell_px + minx
    cy = (np.arange(ny) + 0.5) * cell_px + miny
    CX, CY = np.meshgrid(cx, cy)
    centers = np.column_stack([CX.ravel(), CY.ravel()])
    inside_cell = path.contains_points(centers, radius=EDGE_TOL).reshape(ny, nx)
    denom = int(inside_cell.sum())
    if denom == 0:
        return 0.0, n_in, mnn, sig

    r_px = max(DILATE_R_MIN, DILATE_R_SCALE_MNN * mnn, DILATE_R_SCALE_SIG * sig)
    rcell = int(np.ceil(r_px / cell_px))
    rcell = int(min(rcell, MAX_DILATE_CELLS))
    struct = _disk_struct(rcell)
    occ_dil = binary_dilation(occ, structure=struct)

    cov_grid = float((occ_dil & inside_cell).sum()) / float(denom)
    return cov_grid, n_in, mnn, sig

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

def coverage_ultrafast(box, points_pix, grid, cell):
    poly_xy = np.array([(p['x'], p['y']) for p in box], dtype=float)
    poly = Polygon(poly_xy)
    area = poly.area

    cov_grid, n_in, mnn, sig = coverage_fast_grid(box, points_pix, grid, cell)
    if n_in < MIN_POINTS_IN_BOX or area <= 0:
        return 0.0, n_in, None, False, sig

    cov = cov_grid
    used_refine = False
    if abs(cov_grid - COVERAGE_THRESH) < REFINE_BAND:
        minx, miny = poly_xy.min(axis=0); maxx, maxy = poly_xy.max(axis=0)
        cand_idx = query_grid(grid, cell, (minx, miny, maxx, maxy))
        cand = points_pix[cand_idx]
        path = Path(poly_xy)
        in_mask = path.contains_points(cand, radius=EDGE_TOL)
        pts_in = cand[in_mask]
        cov_alpha = coverage_refine_alpha(box, pts_in, poly, area, mnn, sig)
        cov = max(cov_grid, cov_alpha)
        used_refine = True

    return float(min(max(cov, 0.0), 1.0)), n_in, (1.0/(2.0*mnn) if mnn>0 else None), used_refine, float(sig)

def find_matching_las(base_name):
    m = re.search(r'(\d+)$', base_name)
    candidates = []
    if m:
        idx = m.group(1)
        candidates.append(os.path.join(las_folder, f"{idx}_point_cloud.las"))
    candidates.append(os.path.join(las_folder, f"{base_name}_point_cloud.las"))
    for p in candidates:
        if os.path.exists(p): return p
    return None

def save_preview_fast(tif_path, boxes_data, survivors_idx, survivors_f2_idx, pix, out_png):
    with rasterio.open(tif_path) as ds:
        W, H = ds.width, ds.height
        scale = min(1.0, PREVIEW_MAX_DIM / max(W, H))
        out_h = max(1, int(H * scale))
        out_w = max(1, int(W * scale))

        if ds.count >= 3:
            arr = ds.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
            arr = np.moveaxis(arr, 0, 2)  # (H,W,3)
        else:
            a1 = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            arr = np.stack([a1, a1, a1], axis=2)

    if arr.dtype != np.uint8:
        vmin = float(np.percentile(arr, 2))
        vmax = float(np.percentile(arr, 98))
        denom = max(vmax - vmin, 1e-6)
        arr = np.clip((arr - vmin) * (255.0 / denom), 0, 255).astype(np.uint8)

    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    def _draw_box(coords, color, width):
        draw.polygon([(p['x'] * scale, p['y'] * scale) for p in coords], outline=color, width=width)

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

# ===================== MAIN LOOP =====================
tif_files = [f for f in os.listdir(tif_folder) if f.lower().endswith('.tif')]
tif_files.sort()

for tif_file in tif_files:
    base_name = os.path.splitext(tif_file)[0]
    tif_path = os.path.join(tif_folder, tif_file)
    json_path = os.path.join(json_folder, f"{base_name}_lines.json")
    las_path = find_matching_las(base_name)

    if not os.path.exists(json_path) or las_path is None:
        print(f"[SKIP] {base_name}: missing JSON or LAS. json={os.path.exists(json_path)}, las={las_path is not None}")
        continue

    # Read TIFF 
    with rasterio.open(tif_path) as tif:
        T = tif.transform; W, H = tif.width, tif.height; crs = tif.crs

    # Read boxes
    with open(json_path, 'r') as f:
        boxes_data = json.load(f)
    if not boxes_data:
        print(f"[WARN] {base_name}: boxes empty.")
        continue

    # Read LAS & to pixel
    las = laspy.read(las_path)
    xy = np.column_stack([las.x, las.y])
    cols, rows = (~T) * (xy[:, 0], xy[:, 1])
    inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    pix = np.column_stack([cols[inside], rows[inside]])

    print(f"\n=== {base_name} ===")
    print(f"TIFF: {W}x{H}, CRS={crs}, LAS in-image: {inside.sum()} / {len(xy)} ({inside.mean():.2%}), boxes: {len(boxes_data)}")

    # Grid 
    grid, cell = build_grid(pix, GRID_CELL_PX)

    # Precompute polygons (F2)
    polys = [Polygon([(p["x"], p["y"]) for p in box]) for box in boxes_data]
    valid_mask = [poly.is_valid and (poly.area > 0) for poly in polys]

    # ---------- F1: coverage ----------
    survivors, survivors_idx = [], []
    kept, dropped_lowpts, dropped_cov = 0, 0, 0
    box_logs = []

    for i, box in enumerate(boxes_data):
        if not valid_mask[i]:
            dropped_cov += 1
            box_logs.append((i, 0, 0.0, 0.0, None, False, 0.0))
            continue

        cov, n_in, a_used, refined, sig = coverage_ultrafast(box, pix, grid, cell)
        area = float(polys[i].area)

        if n_in < MIN_POINTS_IN_BOX:
            dropped_lowpts += 1
        elif cov >= COVERAGE_THRESH:
            survivors.append(box); survivors_idx.append(i); kept += 1
        else:
            dropped_cov += 1

        box_logs.append((i, n_in, cov, area, a_used, refined, sig))

    # ---------- F2----------
    valid_idx  = [i for i, vm in enumerate(valid_mask) if vm]
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
            if cands is None: return []
            if isinstance(cands, np.ndarray) and np.issubdtype(cands.dtype, np.integer):
                return [valid_idx[int(k)] for k in cands.tolist()]
            out = []
            for g in cands:
                if g is not None: out.extend(wkb2orig.get(g.wkb, []))
            return out

        def edge_neighbors(i):
            Pi = polys[i]
            try:
                cands = tree.query(Pi, predicate="touches")
            except TypeError:
                cands = tree.query(Pi)  # shapely 1.x fallback
            cand_idx = _normalize_candidate_indices(cands)
            nei = []
            for j in cand_idx:
                if j == i or not valid_mask[j]: continue
                inter = Pi.intersection(polys[j])
                if inter.length > 1e-9 and inter.geom_type in ("LineString","LinearRing","MultiLineString"):
                    nei.append(j)
            return nei

        for i in dropped_idx:
            nei = edge_neighbors(i)
            if nei and all((j in kept_set) for j in nei):
                survivors_f2_idx.append(i)

    survivors_f2 = [boxes_data[i] for i in survivors_f2_idx]
    final_boxes = survivors + survivors_f2

    # ---------- SAVE ----------
    out_json = os.path.join(output_folder, f"{base_name}_final_boxes_data.json")
    with open(out_json, 'w') as f:
        json.dump(final_boxes, f, indent=2)
    print(f"[SAVE] F1: {len(survivors)} | F2: {len(survivors_f2)} | drop_lowpts={dropped_lowpts}, drop_cov={dropped_cov} → {out_json}")

    # ---------- PREVIEW----------
    out_png = os.path.join(output_folder, f"{base_name}_boxes_visualization.png")
    save_preview_fast(tif_path, boxes_data, survivors_idx, survivors_f2_idx, pix, out_png)
    print(f"[SAVE] preview: {out_png}")

    box_logs.sort(key=lambda t: t[1], reverse=True)
    print("Top boxes by points_in:")
    for idx, n_in, cov, area, a, refined, sig in box_logs[:10]:
        print(f"  Box {idx:4d}  pts_in={n_in:6d}  coverage={cov:6.3f}  area={area:.1f}  "
              f"alpha={('%.4f'%a) if a is not None else 'NA'}  refined={refined}  sigma_minor={sig:.3f}")
