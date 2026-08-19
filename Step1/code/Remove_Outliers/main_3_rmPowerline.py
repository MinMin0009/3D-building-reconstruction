import os, glob, time, json, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from Tree_Segmentation.config import load_config

config = load_config(None)
result_file_path = config['paths']['result_path']
parent_directory = result_file_path

ground_dir = os.path.join(parent_directory, 'ground_las(9)')
non_ground_dir = os.path.join(parent_directory, 'non_ground_las(9)')
ground_tif_dir = os.path.join(parent_directory, 'ground_tiff(9)')
non_ground_tif_dir = os.path.join(parent_directory, 'non_ground_tiff(9)')
debug_dir = os.path.join(parent_directory, '_debug_main2')
out_dir = os.path.join(parent_directory, 'output_new_npy')
os.makedirs(out_dir, exist_ok=True)

P = config.get('params', {})
KEEP_AGL_THR = float(P.get('keep_agl_threshold', 3.0))
SAVE_QC = bool(P.get('save_qc', True))
TREAT_ZERO_AS_NODATA = bool(P.get('treat_zero_as_nodata', False))
SENTINEL = -9.969e6
POWERLINE_REMOVE_ENABLE = bool(P.get('powerline_remove_enable', True))
POWERLINE_BUFFER_PX = int(P.get('powerline_buffer_px', 3))

def log(msg): print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg)

def fill_nearest(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    valid = np.isfinite(out)
    if TREAT_ZERO_AS_NODATA: valid &= (out != 0)
    if valid.all(): return out
    _, (ii, jj) = distance_transform_edt(~valid, return_indices=True)
    out[~valid] = out[ii[~valid], jj[~valid]]
    return out

def load_stats(tag: str, part_i: int):
    js = os.path.join(debug_dir, f"{tag}_part_{part_i}.stats.json")
    if not os.path.isfile(js): return None
    try:
        with open(js, "r", encoding="utf-8") as f:
            info = json.load(f)
        dx = float(info.get('est_spacing', {}).get('dx'))
        dy = float(info.get('est_spacing', {}).get('dy'))
        return dx, dy
    except Exception:
        return None

def reproject_to_tif(src_arr, src_transform, dst_shape, dst_transform):
    dst = np.full(dst_shape, SENTINEL, dtype=np.float32)
    src = src_arr.astype(np.float32, copy=True)
    src[~np.isfinite(src)] = SENTINEL
    reproject(source=src, destination=dst, src_transform=src_transform, dst_transform=dst_transform, src_crs=None, dst_crs=None, src_nodata=SENTINEL, dst_nodata=SENTINEL, resampling=Resampling.nearest)
    dst[dst == SENTINEL] = np.nan
    return dst

def list_parts():
    patterns = [
        os.path.join(ground_dir, 'ground_point_cloud_part_*.native.npy'),
        os.path.join(ground_dir, 'ground_point_cloud_part_*.npy'),
        os.path.join(non_ground_dir, 'non_ground_point_cloud_part_*.native.npy'),
        os.path.join(non_ground_dir, 'non_ground_point_cloud_part_*.npy'),
    ]
    parts = set()
    for pat in patterns:
        for p in glob.glob(pat):
            m = re.search(r'part_(\d+)', os.path.basename(p))
            if m: parts.add(int(m.group(1)))
    return sorted(parts) if parts else list(range(1, 10))

def remove_powerlines_by_cooccupancy(out_tif: np.ndarray, ground_tif_aligned: np.ndarray, buffer_px: int = 3):
    if out_tif is None or ground_tif_aligned is None: return out_tif, 0
    if out_tif.shape != ground_tif_aligned.shape: return out_tif, 0
    binary_ground = np.isfinite(ground_tif_aligned)
    if TREAT_ZERO_AS_NODATA: binary_ground &= (ground_tif_aligned != 0)
    binary_out = np.isfinite(out_tif)
    if TREAT_ZERO_AS_NODATA: binary_out &= (out_tif != 0)
    mask = binary_ground & binary_out
    if buffer_px > 0: mask = binary_dilation(mask, iterations=buffer_px)
    out2 = out_tif.copy()
    removed = int(np.count_nonzero(np.isfinite(out2) & mask))
    out2[mask] = np.nan
    return out2, removed

def main():
    t0 = time.time()
    log("Start main_3")
    parts = list_parts()
    log(f"Parts: {parts}")

    for i in parts:
        log(f"[Part {i}]")
        g_native = os.path.join(ground_dir, f'ground_point_cloud_part_{i}.native.npy')
        ng_native = os.path.join(non_ground_dir, f'non_ground_point_cloud_part_{i}.native.npy')
        g_npy = os.path.join(ground_dir, f'ground_point_cloud_part_{i}.npy')
        ng_npy = os.path.join(non_ground_dir, f'non_ground_point_cloud_part_{i}.npy')
        g_tif = os.path.join(ground_tif_dir, f'ground_point_cloud_part_{i}.tiff')
        ng_tif = os.path.join(non_ground_tif_dir, f'non_ground_point_cloud_part_{i}.tiff')

        use_native = os.path.isfile(g_native) and os.path.isfile(ng_native)
        tif_transform = None; H_tif = W_tif = None
        tif_for_part = g_tif if os.path.isfile(g_tif) else (ng_tif if os.path.isfile(ng_tif) else None)
        if use_native and tif_for_part:
            with rasterio.open(tif_for_part) as src:
                tif_transform = src.transform
                H_tif, W_tif = src.height, src.width
        elif use_native and not tif_for_part:
            use_native = False

        if use_native:
            g_arr = np.load(g_native).astype(np.float32)
            ng_arr = np.load(ng_native).astype(np.float32)
            sp = load_stats('ground', i) or load_stats('non_ground', i)
            if sp is None:
                dx = float(abs(tif_transform.a)); dy = float(abs(tif_transform.e))
            else:
                dx, dy = sp
            left, top = tif_transform.c, tif_transform.f
            native_transform = from_origin(left, top, dx, dy)
            g_fill = fill_nearest(g_arr)
            ng_ok = np.isfinite(ng_arr) & (ng_arr != 0 if TREAT_ZERO_AS_NODATA else True)
            g_ok = np.isfinite(g_fill) & (g_fill != 0 if TREAT_ZERO_AS_NODATA else True)
            ok = ng_ok & g_ok
            agl_native = np.full_like(ng_arr, np.nan, dtype=np.float32)
            agl_native[ok] = (ng_arr[ok] - g_fill[ok]).astype(np.float32)
            keep_native = np.isfinite(agl_native) & (agl_native >= KEEP_AGL_THR)
            out_native = np.full_like(ng_arr, np.nan, dtype=np.float32)
            out_native[keep_native] = ng_arr[keep_native].astype(np.float32)
            out = reproject_to_tif(out_native, native_transform, (H_tif, W_tif), tif_transform)
            if POWERLINE_REMOVE_ENABLE:
                g_fill_tif = reproject_to_tif(g_fill, native_transform, (H_tif, W_tif), tif_transform)
                out, removed_pwl = remove_powerlines_by_cooccupancy(out, g_fill_tif, POWERLINE_BUFFER_PX)
                log(f"removed powerline: {removed_pwl}")
            kept_px = int(np.count_nonzero(np.isfinite(out)))
            log(f"kept_px(native->tif): {kept_px}")
        else:
            if not (os.path.isfile(g_npy) and os.path.isfile(ng_npy)):
                log("missing TIF-aligned npy, skip"); continue
            g_arr = np.load(g_npy).astype(np.float32)
            ng_arr = np.load(ng_npy).astype(np.float32)
            if g_arr.shape != ng_arr.shape:
                log(f"shape mismatch {g_arr.shape} vs {ng_arr.shape}, skip"); continue
            g_fill = fill_nearest(g_arr)
            ng_ok = np.isfinite(ng_arr) & (ng_arr != 0 if TREAT_ZERO_AS_NODATA else True)
            g_ok = np.isfinite(g_fill) & (g_fill != 0 if TREAT_ZERO_AS_NODATA else True)
            ok = ng_ok & g_ok
            agl = np.full_like(ng_arr, np.nan, dtype=np.float32)
            agl[ok] = (ng_arr[ok] - g_fill[ok]).astype(np.float32)
            keep = np.isfinite(agl) & (agl >= KEEP_AGL_THR)
            out = np.full_like(ng_arr, np.nan, dtype=np.float32)
            out[keep] = ng_arr[keep].astype(np.float32)
            if POWERLINE_REMOVE_ENABLE:
                out, removed_pwl = remove_powerlines_by_cooccupancy(out, g_fill, POWERLINE_BUFFER_PX)
                log(f"removed powerline: {removed_pwl}")
            kept_px = int(np.count_nonzero(np.isfinite(out)))
            log(f"kept_px: {kept_px}")

        out_path = os.path.join(out_dir, f'rm_car_point_cloud_part_{i}.npy')
        np.save(out_path, out.astype(np.float32))
        log(f"saved: {out_path}")

        if SAVE_QC:
            try:
                step = max(1, int(round(max(out.shape) / 2000)))
                sl = (slice(None, None, step), slice(None, None, step))
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(fill_nearest(g_arr if not use_native else g_arr)[sl], interpolation='nearest')
                axs[0].set_title('Ground (filled)')
                axs[1].imshow((agl if not use_native else agl_native)[sl], interpolation='nearest')
                axs[1].set_title('AGL')
                im2 = axs[2].imshow(out[sl], interpolation='nearest')
                axs[2].set_title(f'Keep AGL ≥ {KEEP_AGL_THR} + Remove Powerline')
                fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
                for ax in axs: ax.set_xticks([]); ax.set_yticks([])
                plt.tight_layout()
                qc_path = os.path.join(out_dir, f'qc_part_{i}.png')
                plt.savefig(qc_path, dpi=140); plt.close(fig)
                log(f"QC: {qc_path}")
            except Exception as e:
                log(f"QC failed: {e}")

    log(f"Done. Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
