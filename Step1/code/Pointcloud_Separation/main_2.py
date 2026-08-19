
import os, json, time, math
import numpy as np
import laspy, CSF
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from Tree_Segmentation.config import load_config

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# =========== CONFIG & PATHS ===========
config      = load_config(None)
tif_path    = config['paths']['tiff_path']
result_path = config['paths']['result_path']
parent_dir  = result_path

ground_dir  = os.path.join(parent_dir, 'ground_las(9)')
nong_dir    = os.path.join(parent_dir, 'non_ground_las(9)')
gtif_dir    = os.path.join(parent_dir, 'ground_tiff(9)')      
ntif_dir    = os.path.join(parent_dir, 'non_ground_tiff(9)') 
os.makedirs(ground_dir, exist_ok=True)
os.makedirs(nong_dir,   exist_ok=True)
os.makedirs(gtif_dir,   exist_ok=True)
os.makedirs(ntif_dir,   exist_ok=True)

# Rasterizing directly to a 2 cm orthophoto grid can leave roof interiors sparse
# when point spacing is larger than the pixel size. A small adaptive point support
# keeps roofs continuous without returning to the huge native LAS grid.
MAX_SPACING_SAMPLE_POINTS = 100_000
SPLAT_RADIUS_FACTOR = 0.75
MAX_SPLAT_RADIUS_PX = 5
MIN_NONGROUND_SPLAT_RADIUS_PX = 2
FULL_PREVIEW_WIDTH = 3600

def log(msg): print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg)

# ---------- helpers ----------
def np_array(a, dtype=None):
    return np.asarray(a, dtype=dtype) if dtype is not None else np.asarray(a)

def estimate_spacing(v_like, round_mm=3):
    v = np_array(v_like, dtype=np.float64)
    if v.size < 2: return None
    u = np.unique(np.round(v, round_mm))
    if u.size < 2: return None
    d = np.diff(np.sort(u))
    d = d[(d > 1e-4) & (d < np.percentile(d, 99))]
    return float(np.median(d)) if d.size else None

def estimate_xy_spacing(X, Y, max_sample=MAX_SPACING_SAMPLE_POINTS):
    X = np_array(X, dtype=np.float64)
    Y = np_array(Y, dtype=np.float64)
    if X.size < 2:
        return None

    if X.size > max_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.size, max_sample, replace=False)
        pts = np.column_stack((X[idx], Y[idx]))
    else:
        pts = np.column_stack((X, Y))

    try:
        dist, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    except TypeError:
        dist, _ = cKDTree(pts).query(pts, k=2)

    nn = dist[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return None
    return {
        "median": float(np.median(nn)),
        "q75": float(np.percentile(nn, 75)),
        "q90": float(np.percentile(nn, 90)),
    }

def choose_splat_radius_px(X, Y, pixel_size, tag):
    if "non_ground" not in tag:
        return 0, None

    spacing = estimate_xy_spacing(X, Y)
    if spacing is None:
        return MIN_NONGROUND_SPLAT_RADIUS_PX, None

    radius = int(math.ceil((spacing["q75"] * SPLAT_RADIUS_FACTOR) / pixel_size))
    radius = max(MIN_NONGROUND_SPLAT_RADIUS_PX, min(MAX_SPLAT_RADIUS_PX, radius))
    return radius, spacing

def rasterize_native(X, Y, Z, bounds, dx, dy, agg='mean'):
    X = np_array(X, dtype=np.float64)
    Y = np_array(Y, dtype=np.float64)
    Z = np_array(Z, dtype=np.float32)

    x0, x1 = bounds.left, bounds.right
    y0, y1 = bounds.bottom, bounds.top
    W = int(math.ceil((x1 - x0) / dx))
    H = int(math.ceil((y1 - y0) / dy))
    if W <= 0 or H <= 0:
        return np.full((0,0), np.nan, dtype=np.float32), from_origin(x0, y1, dx, dy)

    cols = np.floor((X - x0) / dx).astype(np.int64)
    rows = np.floor((y1 - Y) / dy).astype(np.int64)
    sel  = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    if not np.any(sel):
        return np.full((H, W), np.nan, dtype=np.float32), from_origin(x0, y1, dx, dy)

    cols = cols[sel]; rows = rows[sel]; z = Z[sel]
    lin = rows * W + cols

    if agg == 'min':
        grid = np.full((H, W), np.inf, dtype=np.float32)
        flat = grid.ravel()
        np.minimum.at(flat, lin, z)
        grid[grid == np.inf] = np.nan
    elif agg == 'max':
        grid = np.full((H, W), -np.inf, dtype=np.float32)
        flat = grid.ravel()
        np.maximum.at(flat, lin, z)
        grid[grid == -np.inf] = np.nan
    else:  # mean
        sum_ = np.zeros(H*W, dtype=np.float64)
        cnt_ = np.zeros(H*W, dtype=np.int32)
        np.add.at(sum_, lin, z)
        np.add.at(cnt_, lin, 1)
        grid = np.full((H, W), np.nan, dtype=np.float32)
        m = cnt_ > 0
        grid.ravel()[m] = (sum_[m] / cnt_[m]).astype(np.float32)

    transform = from_origin(x0, y1, dx, dy)
    return grid, transform

def rasterize_to_tif_grid(X, Y, Z, out_shape, out_transform, agg='mean', splat_radius_px=0):
    X = np_array(X, dtype=np.float64)
    Y = np_array(Y, dtype=np.float64)
    Z = np_array(Z, dtype=np.float32)

    H, W = out_shape
    if W <= 0 or H <= 0:
        return np.full((0, 0), np.nan, dtype=np.float32)

    dx = float(out_transform.a)
    dy = float(abs(out_transform.e))
    x0 = float(out_transform.c)
    y1 = float(out_transform.f)

    cols = np.floor((X - x0) / dx).astype(np.int64)
    rows = np.floor((y1 - Y) / dy).astype(np.int64)
    sel = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)

    grid = np.full((H, W), np.nan, dtype=np.float32)
    if not np.any(sel):
        return grid

    cols = cols[sel]
    rows = rows[sel]
    z = Z[sel]
    flat = grid.ravel()

    if agg == 'min':
        flat[:] = np.inf
        for dr, dc in splat_offsets(splat_radius_px):
            rr = rows + dr
            cc = cols + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if np.any(ok):
                np.minimum.at(flat, rr[ok] * W + cc[ok], z[ok])
        flat[flat == np.inf] = np.nan
    elif agg == 'max':
        flat[:] = -np.inf
        for dr, dc in splat_offsets(splat_radius_px):
            rr = rows + dr
            cc = cols + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if np.any(ok):
                np.maximum.at(flat, rr[ok] * W + cc[ok], z[ok])
        flat[flat == -np.inf] = np.nan
    else:
        lin = rows * W + cols
        sum_ = np.zeros(H * W, dtype=np.float64)
        cnt_ = np.zeros(H * W, dtype=np.int32)
        np.add.at(sum_, lin, z)
        np.add.at(cnt_, lin, 1)
        m = cnt_ > 0
        flat[m] = (sum_[m] / cnt_[m]).astype(np.float32)

    return grid

def splat_offsets(radius_px):
    if radius_px <= 0:
        return [(0, 0)]

    out = []
    r2 = radius_px * radius_px
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc <= r2:
                out.append((dr, dc))
    return out

def debug_plots(mask, tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if plt is None:
        preview = (mask.astype(np.uint8) * 255)
        Image.fromarray(preview, mode='L').save(os.path.join(out_dir, f'dbg_mask_{tag}.png'))
        return

    # mask
    fig = plt.figure(figsize=(6,7))
    plt.imshow(mask, interpolation='nearest')
    plt.title(f'Valid mask ({tag})'); plt.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'dbg_mask_{tag}.png'), dpi=150)
    plt.close(fig)
    # hist
    vpc = np.sum(mask, axis=0)
    vpr = np.sum(mask, axis=1)
    fig2, ax = plt.subplots(2,1, figsize=(12,6))
    ax[0].plot(vpc); ax[0].set_title(f'#valid per COLUMN ({tag})'); ax[0].set_ylabel('count')
    ax[1].plot(vpr); ax[1].set_title(f'#valid per ROW ({tag})');    ax[1].set_ylabel('count')
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, f'dbg_hist_{tag}.png'), dpi=150)
    plt.close(fig2)

def fill_nearest_for_preview(arr):
    valid = np.isfinite(arr)
    if valid.all() or not np.any(valid):
        return arr
    _, (ii, jj) = distance_transform_edt(~valid, return_indices=True)
    out = arr.copy()
    out[~valid] = out[ii[~valid], jj[~valid]]
    return out

def downsample_npy_for_preview(npy_path, out_h, out_w, agg='max'):
    arr = np.load(npy_path, mmap_mode='r')
    H, W = arr.shape
    out = np.full((out_h, out_w), np.nan, dtype=np.float32)

    row_edges = np.linspace(0, H, out_h + 1, dtype=np.int64)
    col_edges = np.linspace(0, W, out_w + 1, dtype=np.int64)

    for rr in range(out_h):
        r0, r1 = row_edges[rr], row_edges[rr + 1]
        if r1 <= r0:
            r1 = min(H, r0 + 1)
        for cc in range(out_w):
            c0, c1 = col_edges[cc], col_edges[cc + 1]
            if c1 <= c0:
                c1 = min(W, c0 + 1)

            block = np.asarray(arr[r0:r1, c0:c1])
            valid = np.isfinite(block)
            if not np.any(valid):
                continue
            vals = block[valid]
            if agg == 'min':
                out[rr, cc] = np.min(vals)
            elif agg == 'mean':
                out[rr, cc] = np.mean(vals)
            else:
                out[rr, cc] = np.max(vals)

    return out

def save_height_preview_png(full_arr, out_png, title):
    valid = np.isfinite(full_arr)
    if not np.any(valid):
        img = Image.new('RGB', (full_arr.shape[1], full_arr.shape[0]), (0, 0, 0))
    else:
        filled = fill_nearest_for_preview(full_arr)
        lo, hi = np.nanpercentile(full_arr[valid], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(full_arr[valid])), float(np.nanmax(full_arr[valid]))
        norm = np.clip((filled - lo) / max(hi - lo, 1e-6), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb = np.dstack((gray, gray, gray))
        rgb[~valid] = (0, 0, 0)
        img = Image.fromarray(rgb, mode='RGB')

    draw = ImageDraw.Draw(img)
    text = f"{title} | valid={int(np.count_nonzero(valid)):,}/{valid.size:,}"
    draw.rectangle((8, 8, min(img.width - 8, 980), 42), fill=(0, 0, 0))
    draw.text((16, 17), text, fill=(255, 255, 255))
    img.save(out_png)

def save_full_area_main2_previews(quads):
    if not quads:
        return

    tile_h, tile_w = quads[0][4]
    scale = (tile_w * 3) / FULL_PREVIEW_WIDTH
    preview_tile_w = max(1, int(round(tile_w / scale)))
    preview_tile_h = max(1, int(round(tile_h / scale)))
    preview_h = preview_tile_h * 3
    preview_w = preview_tile_w * 3

    for prefix, out_dir, agg in [
        ("ground_point_cloud_part", ground_dir, "min"),
        ("non_ground_point_cloud_part", nong_dir, "max"),
    ]:
        full = np.full((preview_h, preview_w), np.nan, dtype=np.float32)
        for idx in range(1, 10):
            npy_path = os.path.join(out_dir, f"{prefix}_{idx}.npy")
            if not os.path.isfile(npy_path):
                continue

            tile = downsample_npy_for_preview(npy_path, preview_tile_h, preview_tile_w, agg=agg)
            r = (idx - 1) // 3
            c = (idx - 1) % 3
            full[
                r * preview_tile_h:(r + 1) * preview_tile_h,
                c * preview_tile_w:(c + 1) * preview_tile_w,
            ] = tile

        out_png = os.path.join(parent_dir, f"{prefix}_full_area_interpolated_preview.png")
        save_height_preview_png(full, out_png, prefix)
        log(f"Full-area interpolated preview saved: {out_png}")

def reproject_to_tif(src_arr, src_transform, dst_shape, dst_transform, crs):
    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.average
    )
    return dst

def run_csf(las_path):
    inFile = laspy.read(las_path)
    pts = inFile.points
    xyz = np.vstack((np_array(inFile.x), np_array(inFile.y), np_array(inFile.z))).T

    csf = CSF.CSF()
    csf.params.bSloopSmooth    = True
    csf.params.cloth_resolution= 1.5
    csf.params.time_step       = 0.65
    csf.params.class_threshold = 0.5
    csf.params.interations     = 500
    csf.params.rigidness       = 3
    csf.setPointCloud(xyz)

    g_idx = CSF.VecInt(); n_idx = CSF.VecInt()
    csf.do_filtering(g_idx, n_idx)

    g = laspy.LasData(inFile.header); g.points = pts[np.array(g_idx)]
    n = laspy.LasData(inFile.header); n.points = pts[np.array(n_idx)]
    return g, n

# =========== MAIN ===========
def main():
    t0 = time.time()

    tif_name   = os.path.splitext(os.path.basename(tif_path))[0]
    las_dir    = os.path.dirname(config['paths']['las_path'])
    las_path   = os.path.join(las_dir, f"{tif_name}_filtered.las") 

    log("Reading & splitting TIF...")
    with rasterio.open(tif_path) as src:
        tr_all  = src.transform
        bounds  = src.bounds
        width, height = src.width, src.height
        crs     = src.crs
        # chia 3x3 theo bounds
        third_x = (bounds.right - bounds.left) / 3.0
        third_y = (bounds.top   - bounds.bottom) / 3.0
        quads = []
        for r in range(3):
            for c in range(3):
                left   = bounds.left + c*third_x
                right  = left + third_x
                top    = bounds.top - r*third_y
                bottom = top - third_y
                col_off = int((left - bounds.left) / tr_all.a)
                row_off = int((bounds.top - top)  / -tr_all.e)
                w = int((right - left)  / tr_all.a)
                h = int((top   - bottom)/ -tr_all.e)
                win = Window(col_off, row_off, w, h)
                t_win = rasterio.windows.transform(win, tr_all)
                quads.append((left, right, bottom, top, (h, w), t_win, crs))

    log("Running CSF (ground / non-ground)...")
    ground_las, nong_las = run_csf(las_path)

    log("Saving parts & rasterizing with DEBUG...")
    for i, (left, right, bottom, top, (h_tif, w_tif), t_win, crs) in enumerate(quads, start=1):
        def cut(las_obj):
            X = np_array(las_obj.x); Y = np_array(las_obj.y); Z = np_array(las_obj.z)
            m = (X >= left) & (X < right) & (Y >= bottom) & (Y < top)
            return X[m], Y[m], Z[m]

        for tag, las_obj, out_dir, agg in [
            (f"ground_point_cloud_part_{i}",     ground_las, ground_dir, 'min'),
            (f"non_ground_point_cloud_part_{i}", nong_las,   nong_dir,   'max'),
        ]:
            X, Y, Z = cut(las_obj)

            if X.size == 0:
                arr_tif = np.full((h_tif, w_tif), np.nan, dtype=np.float32)
                np.save(os.path.join(out_dir, f"{tag}.npy"), arr_tif)
                log(json.dumps({"tag": tag, "empty": True}))
                continue

            dx = float(t_win.a)
            dy = float(abs(t_win.e))
            splat_radius_px, xy_spacing = choose_splat_radius_px(X, Y, max(dx, dy), tag)
            arr_tif = rasterize_to_tif_grid(
                X, Y, Z, (h_tif, w_tif), t_win, agg=agg, splat_radius_px=splat_radius_px
            )

            npy_path = os.path.join(out_dir, f"{tag}.npy")
            np.save(npy_path, arr_tif.astype(np.float32))

            np.save(os.path.join(out_dir, f"{tag}.native.npy"), arr_tif.astype(np.float32))
            valid = np.isfinite(arr_tif)
            debug_plots(valid, tag, out_dir)

            info = {
                "tag": tag,
                "tif_px": {"x": float(tr_all.a), "y": float(abs(tr_all.e))},
                "grid_spacing": {"dx": dx, "dy": dy, "source": "tif_window"},
                "xy_spacing": xy_spacing,
                "splat_radius_px": splat_radius_px,
                "valid_ratio": float(np.count_nonzero(valid)) / valid.size if valid.size else 0.0,
                "non_empty_cols": int(np.count_nonzero(np.sum(valid, axis=0) > 0)),
                "non_empty_rows": int(np.count_nonzero(np.sum(valid, axis=1) > 0)),
            }
            log(json.dumps(info))
            print(f"  saved NPY: {npy_path}")

    log("Saving full-area interpolated previews from main_2 NPY outputs...")
    save_full_area_main2_previews(quads)

    log(f"Done main_2_native_debug in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
