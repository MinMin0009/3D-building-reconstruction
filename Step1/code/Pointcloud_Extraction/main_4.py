import os, time, gc
import numpy as np
import pandas as pd
import rasterio
import laspy
import matplotlib.pyplot as plt
from joblib import dump, load
from scipy.ndimage import (
    convolve, binary_dilation, binary_propagation, binary_closing,
    label, find_objects
)
from sklearn.cluster import DBSCAN

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from Tree_Segmentation.config import load_config

# ---------- utils ----------
def log(msg): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
def ensure_exists(path): os.makedirs(os.path.dirname(path), exist_ok=True)
def make_disk(r:int)->np.ndarray:
    if r<=0: return np.ones((1,1), dtype=bool)
    y,x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x+y*y) <= r*r
def vectorized_geo_to_pixel(xs, ys, inv_affine):
    cols = inv_affine.a*xs + inv_affine.b*ys + inv_affine.c
    rows = inv_affine.d*xs + inv_affine.e*ys + inv_affine.f
    return rows, cols

class DSU:
    def __init__(self,n):
        self.p=np.arange(n,dtype=np.int32)
        self.r=np.zeros(n,dtype=np.int8)
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1

def merge_centroids_faiss(C, eps=3.0, min_samples=1):
    if C.size==0: return np.array([], dtype=np.int32)
    if _HAS_FAISS:
        C=C.astype(np.float32, copy=False)
        index=faiss.IndexFlatL2(C.shape[1]); index.add(C)
        lims,_,I=index.range_search(C, float(eps**2))
        nbr=[I[lims[i]:lims[i+1]].tolist() for i in range(len(C))]
        vis=np.zeros(len(C),dtype=bool)
        lab=-np.ones(len(C),dtype=np.int32); cid=0
        for i in range(len(C)):
            if vis[i]: continue
            vis[i]=True
            if len(nbr[i])<min_samples: continue
            lab[i]=cid; seeds=set(nbr[i]); seeds.discard(i)
            while seeds:
                j=seeds.pop()
                if not vis[j]:
                    vis[j]=True
                    if len(nbr[j])>=min_samples: seeds.update(nbr[j])
                if lab[j]==-1: lab[j]=cid
            cid+=1
        return lab
    else:
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(C).astype(np.int32)

def bbox_overlap_area_expanded(bi,bj,r):
    r1min,r1max,c1min,c1max=bi; r2min,r2max,c2min,c2max=bj
    rmin=max(r1min-r, r2min-r); rmax=min(r1max+r, r2max+r)
    cmin=max(c1min-r, c2min-r); cmax=min(c1max+r, c2max+r)
    oh=rmax-rmin+1; ow=cmax-cmin+1
    return 0 if (oh<=0 or ow<=0) else oh*ow

def bbox_chebyshev_gap(bi,bj):
    r1min,r1max,c1min,c1max=bi; r2min,r2max,c2min,c2max=bj
    if r1max<r2min: dr=r2min-r1max-1
    elif r2max<r1min: dr=r1min-r2max-1
    else: dr=0
    if c1max<c2min: dc=c2min-c1max-1
    elif c2max<c1min: dc=c1min-c2max-1
    else: dc=0
    return max(dr,dc)

# ---------- stripe-heal ----------
def _estimate_stripe_kernel(mask_bool: np.ndarray, max_w:int=31):
    col_idx = np.where(mask_bool.any(axis=0))[0]
    row_idx = np.where(mask_bool.any(axis=1))[0]
    dx = float(np.median(np.diff(col_idx))) if col_idx.size>1 else 0.0
    dy = float(np.median(np.diff(row_idx))) if row_idx.size>1 else 0.0
    horiz = dx >= dy  # True => cần kernel ngang
    w = int(round(dx if horiz else dy))
    if w <= 2: w = 5
    if w > max_w: w = max_w
    if (w % 2)==0: w += 1
    kernel = np.ones((1,w), dtype=bool) if horiz else np.ones((w,1), dtype=bool)
    return kernel, ('h' if horiz else 'v'), w

# ---------- main ----------
def main():
    cfg = load_config(None)

    log("Reading paths from configuration...")
    tiff_file = cfg['paths']['tiff_path']
    las_file  = cfg['paths']['las_path']
    result_dir = cfg['paths']['result_path']
    out_npy_dir = os.path.join(result_dir, 'output_new_npy')
    save_path = os.path.join(out_npy_dir, 'cluster_building.pkl')

    # ===== Params =====
    EPS1=12; MIN_SAMPLES1=10; SIZE_THR=10000
    CLOSE_R=1; USE_PROP=True
    JOIN_R=12

    AUTO_STRIPE_HEAL = True
    STRIPE_MAX_W     = 31

    # Seam patch
    SEAM_W = 32
    SEAM_CLOSE_R = 16

    FAISS_EPS=3.0

    # ===== TIF =====
    log("Reading the TIF file...")
    with rasterio.open(tiff_file) as src:
        transform = src.transform
        width, height = src.width, src.height
    inv_transform = ~transform

    # ===== LAS =====
    log("Reading LAS file...")
    las = laspy.read(las_file)
    log("Processing point cloud (vectorized arrays)...")
    x = las.X*las.header.x_scale + las.header.x_offset
    y = las.Y*las.header.y_scale + las.header.y_offset
    z = las.Z*las.header.z_scale + las.header.z_offset
    N = x.shape[0]

    # ===== Parts =====
    log("Scanning parts & shapes...")
    part_paths=[os.path.join(out_npy_dir, f'rm_car_point_cloud_part_{i}.npy') for i in range(1,10)]
    for p in part_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing npy input: {p}")
    parts  = [np.load(p, mmap_mode='r') for p in part_paths]
    shapes = np.array([a.shape for a in parts], dtype=np.int64) # (H,W) 9 part

    # offsets: 3x3 grid
    row_heights,row_widths=[],[]
    for r in range(3):
        r_parts=shapes[r*3:(r+1)*3]
        if len(set(r_parts[:,0].tolist()))!=1:
            raise ValueError(f"Row {r} parts must have equal heights, got {r_parts[:,0].tolist()}")
        row_heights.append(int(r_parts[0,0]))
        row_widths.append(int(r_parts[:,1].sum()))
    if len(set(row_widths))!=1:
        raise ValueError(f"All rows must have equal total width, got {row_widths}")

    row_offs=[0,row_heights[0],row_heights[0]+row_heights[1]]
    col_offs=[]
    for r in range(3):
        w1=int(shapes[r*3+0,1]); w2=int(shapes[r*3+1,1])
        col_offs.append([0,w1,w1+w2])

    # ===== Pass #1: per-part -> components + edge bands =====
    log("Per-part density -> propagation -> stripe-heal -> label-by-closing; collect edge bands...")
    disk1=make_disk(EPS1).astype(np.uint8); structure8=np.ones((3,3),dtype=np.uint8)

    comp_coords=[]; comp_boxes=[]; comp_centers=[]; comp_parts=[]
    edge_bands = {}   # pid -> {'L','R','T','B'}

    for i, arr in enumerate(parts, start=1):
        log(f"  Part {i}: start")
        mask_bool=(~np.isnan(arr))
        mask_u8=mask_bool.astype(np.uint8, copy=False)

        counts=convolve(mask_u8, disk1, mode='constant', cval=0)
        core=(mask_bool & (counts>=MIN_SAMPLES1))
        if CLOSE_R>0:
            bridge=make_disk(CLOSE_R).astype(np.uint8)
            mask_boost=binary_closing(mask_bool, structure=bridge)
        else:
            mask_boost=mask_bool

        grown = binary_propagation(core, structure=structure8, mask=mask_boost) if USE_PROP \
                else (binary_dilation(core, structure=disk1) & mask_boost)

        # stripe-heal 
        if AUTO_STRIPE_HEAL:
            k1, ori, w = _estimate_stripe_kernel(mask_bool, max_w=STRIPE_MAX_W)
            heal = binary_closing(grown, structure=k1)
            log(f"    stripe-heal: orientation={ori}, kernel_width={w}")
        else:
            heal = grown

        proxy = binary_closing(heal, structure=make_disk(JOIN_R))
        lbl_close, nlab = label(proxy, structure=structure8)

        H,W = arr.shape
        r_block, c_block=(i-1)//3, (i-1)%3
        r_off, c_off=row_offs[r_block], col_offs[r_block][c_block]

        # map local label -> global comp-id
        local2global = {}
        kept=0

        if nlab>0:
            boxes = find_objects(lbl_close)
            for lab_id in range(1, nlab+1):
                sl=boxes[lab_id-1]
                if sl is None: continue
                region = (lbl_close[sl]==lab_id) & grown[sl]
                if not np.any(region): continue
                if region.sum() < SIZE_THR: continue

                rr,cc = np.where(region)
                rr_g = rr.astype(np.int32) + np.int32(sl[0].start + r_off)
                cc_g = cc.astype(np.int32) + np.int32(sl[1].start + c_off)
                coords = np.stack((rr_g, cc_g), axis=1)

                comp_coords.append(coords)
                comp_centers.append(coords.mean(axis=0, dtype=np.float32))
                comp_boxes.append((int(rr_g.min()), int(rr_g.max()), int(cc_g.min()), int(cc_g.max())))
                comp_parts.append(i-1)
                local2global[lab_id] = len(comp_coords)-1
                kept+=1
        log(f"  Part {i}: kept {kept} components")

        bw = min(SEAM_W, W//2)
        bh = min(SEAM_W, H//2)
        bands = {}

        L = -np.ones((H, bw), dtype=np.int32)
        R = -np.ones((H, bw), dtype=np.int32)
        T = -np.ones((bh, W), dtype=np.int32)
        B = -np.ones((bh, W), dtype=np.int32)

        if nlab>0 and local2global:
            sub = lbl_close[:, :bw]
            labs = np.unique(sub[grown[:, :bw]])
            for lab in labs:
                if lab==0 or lab not in local2global: continue
                L[sub==lab] = local2global[lab]

            sub = lbl_close[:, W-bw:W]
            labs = np.unique(sub[grown[:, W-bw:W]])
            for lab in labs:
                if lab==0 or lab not in local2global: continue
                R[sub==lab] = local2global[lab]

            sub = lbl_close[:bh, :]
            labs = np.unique(sub[grown[:bh, :]])
            for lab in labs:
                if lab==0 or lab not in local2global: continue
                T[sub==lab] = local2global[lab]

            sub = lbl_close[H-bh:H, :]
            labs = np.unique(sub[grown[H-bh:H, :]])
            for lab in labs:
                if lab==0 or lab not in local2global: continue
                B[sub==lab] = local2global[lab]

        bands['L']=L; bands['R']=R; bands['T']=T; bands['B']=B
        edge_bands[i-1] = bands

        del mask_u8,counts,core,mask_boost,grown,proxy
        if nlab>0: del lbl_close
        gc.collect()

    if len(comp_coords)==0:
        log("No components found in all parts. Exiting."); return

    M=len(comp_coords); log(f"Total components before seam merges: {M}")
    dsu = DSU(M)

    # ===== Seam-patch merges + BRIDGES =====
    bridges = []  # dict{ 'comps': list[int], 'coords': np.ndarray[[r,c],...] }

    def _merge_vertical(pid_left, pid_right):
        bands_left  = edge_bands.get(pid_left)
        bands_right = edge_bands.get(pid_right)
        if bands_left is None or bands_right is None: return

        A = bands_left['R']   # H x bw
        B = bands_right['L']  # H x bw
        if A.size==0 or B.size==0: return

        H  = min(A.shape[0], B.shape[0])
        bw = min(A.shape[1], B.shape[1])
        if H==0 or bw==0: return
        A = A[:H, :bw]
        B = B[:H, :bw]

        # patch + closing 
        patch = np.zeros((H, 2*bw), dtype=bool)
        patch[:, :bw] = (A>=0)
        patch[:, bw:] = (B>=0)
        if not patch.any(): return

        patched = binary_closing(patch, structure=make_disk(SEAM_CLOSE_R))
        if not patched.any(): return
        lbl, nlab = label(patched, structure=np.ones((3,3), dtype=np.uint8))
        if nlab==0: return

        # offsets
        r_row = pid_left//3  
        r_off = row_offs[r_row]
        W_left  = shapes[pid_left,1]
        c_off_L = col_offs[r_row][pid_left%3]
        c_off_R = col_offs[r_row][pid_right%3]

        for lab_id in range(1, nlab+1):
            mask_component = (lbl==lab_id)
            left_mask  = mask_component[:, :bw]
            right_mask = mask_component[:, bw:]

            left_ids  = np.unique(A[left_mask & (A>=0)])
            right_ids = np.unique(B[right_mask & (B>=0)])
            if left_ids.size==0 or right_ids.size==0:
                continue

            # union
            comps = set()
            for li in left_ids:
                for rj in right_ids:
                    dsu.union(int(li), int(rj))
                    comps.add(int(li)); comps.add(int(rj))

            # bridge pixels
            new_mask = mask_component & (~patch)
            if not new_mask.any():  
                continue

            rr, cc = np.where(new_mask)
            # map
            rows_global = rr.astype(np.int32) + np.int32(r_off)
            cols_global = np.empty_like(rows_global, dtype=np.int32)
            left_side  = (cc < bw)
            if np.any(left_side):
                cols_global[left_side] = (
                    cc[left_side].astype(np.int32)
                    + np.int32(c_off_L + (W_left - bw))
                )
            right_side = ~left_side
            if np.any(right_side):
                cols_global[right_side] = (
                    (cc[right_side] - bw).astype(np.int32)
                    + np.int32(c_off_R)
                )
            bridges.append({'comps': list(comps),
                            'coords': np.stack((rows_global, cols_global), axis=1)})

    def _merge_horizontal(pid_top, pid_bottom):
        bands_top    = edge_bands.get(pid_top)
        bands_bottom = edge_bands.get(pid_bottom)
        if bands_top is None or bands_bottom is None: return

        A = bands_top['B']      # bh x W
        B = bands_bottom['T']   # bh x W
        if A.size==0 or B.size==0: return

        bh = min(A.shape[0], B.shape[0])
        W  = min(A.shape[1], B.shape[1])
        if bh==0 or W==0: return
        A = A[:bh, :W]
        B = B[:bh, :W]

        patch = np.zeros((2*bh, W), dtype=bool)
        patch[:bh, :] = (A>=0)
        patch[bh:, :] = (B>=0)
        if not patch.any(): return

        patched = binary_closing(patch, structure=make_disk(SEAM_CLOSE_R))
        if not patched.any(): return
        lbl, nlab = label(patched, structure=np.ones((3,3), dtype=np.uint8))
        if nlab==0: return

        # offsets toàn cục
        c_col = pid_top%3
        c_off_T = col_offs[pid_top//3][c_col]
        c_off_B = col_offs[pid_bottom//3][c_col]
        H_top   = shapes[pid_top,0]
        r_off_T = row_offs[pid_top//3]
        r_off_B = row_offs[pid_bottom//3]

        for lab_id in range(1, nlab+1):
            mask_component = (lbl==lab_id)
            top_mask    = mask_component[:bh, :]
            bottom_mask = mask_component[bh:, :]

            top_ids    = np.unique(A[top_mask & (A>=0)])
            bottom_ids = np.unique(B[bottom_mask & (B>=0)])
            if top_ids.size==0 or bottom_ids.size==0:
                continue

            comps = set()
            for ti in top_ids:
                for bj in bottom_ids:
                    dsu.union(int(ti), int(bj))
                    comps.add(int(ti)); comps.add(int(bj))

            new_mask = mask_component & (~patch)
            if not new_mask.any():
                continue

            rr, cc = np.where(new_mask)
            rows_global = np.empty_like(rr, dtype=np.int32)
            cols_global = np.empty_like(rr, dtype=np.int32)

            top_side = (rr < bh)
            if np.any(top_side):
                rows_global[top_side] = (
                    rr[top_side].astype(np.int32)
                    + np.int32(r_off_T + (H_top - bh))
                )
                cols_global[top_side] = cc[top_side].astype(np.int32) + np.int32(c_off_T)

            bot_side = ~top_side
            if np.any(bot_side):
                rows_global[bot_side] = (
                    (rr[bot_side] - bh).astype(np.int32)
                    + np.int32(r_off_B)
                )
                cols_global[bot_side] = cc[bot_side].astype(np.int32) + np.int32(c_off_B)

            bridges.append({'comps': list(comps),
                            'coords': np.stack((rows_global, cols_global), axis=1)})

    log("Running seam-patch merges (vertical seams)...")
    for r in range(3):
        for c in range(2):
            _merge_vertical(r*3 + c, r*3 + (c+1))

    log("Running seam-patch merges (horizontal seams)...")
    for r in range(2):
        for c in range(3):
            _merge_horizontal(r*3 + c, (r+1)*3 + c)

    roots=np.array([dsu.find(i) for i in range(M)], dtype=np.int32)
    uniq=np.unique(roots); log(f"Groups after seam merges: {uniq.size}")

    # ===== DSU + bridge pixels =====
    log("Building clusters dictionary (with seam bridges)...")
    groups = {}
    for i in range(M):
        g = int(roots[i])
        groups.setdefault(g, []).append(i)

    # bridge pixels 
    bridge_by_root = {}
    for br in bridges:
        if not br['comps']: continue
        root0 = dsu.find(br['comps'][0])
        bridge_by_root.setdefault(root0, []).append(br['coords'])

    clusters={}; new_id=0
    for rroot, idxs in groups.items():
        pts = [comp_coords[i] for i in idxs]
        if rroot in bridge_by_root:
            pts.extend(bridge_by_root[rroot])
        # unique
        if pts:
            all_pts = np.vstack(pts).astype(np.int64, copy=False)
            lin = all_pts[:,0]*np.int64(width) + all_pts[:,1]
            uidx = np.unique(lin, return_index=True)[1]
            final = all_pts[uidx].astype(np.int32, copy=False)
        else:
            final = np.zeros((0,2), dtype=np.int32)
        clusters[new_id]=final; new_id+=1

    # ===== Preview =====
    try:
        log("Saving a light preview (sampled) of merged clusters...")
        rng = np.random.default_rng(0)
        plt.figure(figsize=(9,7))
        shown=0; max_show=500_000
        for _,pts in clusters.items():
            pv = pts if pts.shape[0]<=max_show else pts[rng.choice(pts.shape[0],max_show,replace=False)]
            plt.scatter(pv[:,1],pv[:,0],s=0.1); shown+=pv.shape[0]
            if shown>3_000_000: break
        plt.gca().invert_yaxis(); plt.title('Merged Clusters (sampled preview)')
        plt.xlabel('Column'); plt.ylabel('Row')
        prev=os.path.join(out_npy_dir,'merged_clusters_preview.png')
        plt.tight_layout(); plt.savefig(prev,dpi=150); plt.close()
        log(f"Preview saved: {prev}")
    except Exception as e:
        log(f"(warn) Preview failed: {e}")

    # ===== Save clusters =====
    ensure_exists(save_path); log("Saving clustered data using joblib...")
    dump(clusters, save_path, compress=3)

    # ===== LAS filtering =====
    log("Loading clustered data back..."); building_data=load(save_path)

    log("Preparing union of building pixels (linear indices)...")
    lin_arrays=[]
    for _,pix in building_data.items():
        if pix.size:
            lin_arrays.append(pix[:,0].astype(np.int64)*np.int64(width)+pix[:,1].astype(np.int64))
    if not lin_arrays:
        log("No building pixels to filter. Exiting."); return
    all_lin=np.unique(np.concatenate(lin_arrays)); del lin_arrays; gc.collect()
    log(f"Total building pixels (unique lin idx): {all_lin.size:,}")
    all_lin_sorted=np.sort(all_lin); del all_lin; gc.collect()

    log("Mapping LAS points to pixels in chunks...")
    kept_idx_chunks=[]; ptslin_kept_chunks=[]
    chunk=5_000_000
    for s in range(0,N,chunk):
        e=min(N,s+chunk)
        rows_f,cols_f=vectorized_geo_to_pixel(x[s:e],y[s:e],inv_transform)
        rows=np.floor(rows_f).astype(np.int64); cols=np.floor(cols_f).astype(np.int64)
        inb=(rows>=0)&(rows<height)&(cols>=0)&(cols<width)
        if not np.any(inb): continue
        rows=rows[inb]; cols=cols[inb]; lin=rows*np.int64(width)+cols
        idx=np.searchsorted(all_lin_sorted, lin, side='left')
        valid=idx<all_lin_sorted.size
        hit=np.zeros(lin.shape,dtype=bool)
        if np.any(valid): hit[valid]=(all_lin_sorted[idx[valid]]==lin[valid])
        if np.any(hit):
            kept_idx_chunks.append(s + np.nonzero(inb)[0][hit])
            ptslin_kept_chunks.append(lin[hit])
        del rows_f,cols_f,rows,cols,inb,lin,idx,hit; gc.collect()

    if not kept_idx_chunks:
        log("No LAS points inside building clusters. Exiting."); return

    kept_idx=np.concatenate(kept_idx_chunks)
    pts_lin_kept=np.concatenate(ptslin_kept_chunks)
    del kept_idx_chunks, ptslin_kept_chunks; gc.collect()

    log(f"Filtered points count: {kept_idx.size:,}")

    # CSV + LAS (global)
    log("Saving filtered point cloud to CSV...")
    xk=x[kept_idx]; yk=y[kept_idx]; zk=z[kept_idx]
    df=pd.DataFrame({'X':xk,'Y':yk,'Z':zk})
    csv_path=os.path.join(result_dir,'filtered_point_cloud.csv')
    ensure_exists(csv_path); df.to_csv(csv_path,index=False)

    log("Saving filtered LAS...")
    header=laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.offsets=las.header.offsets; header.scales=las.header.scales
    las_out=laspy.LasData(header)
    las_out.X=(xk-las.header.x_offset)/las.header.x_scale
    las_out.Y=(yk-las.header.y_offset)/las.header.y_scale
    las_out.Z=(zk-las.header.z_offset)/las.header.z_scale
    attrs=['intensity','return_number','num_returns','scan_direction_flag',
           'edge_of_flight_line','classification','synthetic','key_point',
           'withheld','user_data','point_source_id']
    for a in attrs:
        if hasattr(las,a):
            setattr(las_out,a, getattr(las,a)[kept_idx])
    las_out_path=os.path.join(result_dir,'LAS','filtered_point_cloud1.las')
    ensure_exists(las_out_path); las_out.write(las_out_path)

    # per-building
    log("Exporting per-building LAS/CSV...")
    order=np.argsort(pts_lin_kept, kind='mergesort')
    pts_sorted=pts_lin_kept[order]
    x_sorted=xk[order]; y_sorted=yk[order]; z_sorted=zk[order]
    kept_idx_ordered=kept_idx[order]
    csv_dir=os.path.join(result_dir,'CSV'); las_dir=os.path.join(result_dir,'LAS')
    os.makedirs(csv_dir, exist_ok=True); os.makedirs(las_dir, exist_ok=True)

    for bid,pix in building_data.items():
        if pix.size==0: continue
        lin=np.unique(pix[:,0].astype(np.int64)*np.int64(width) + pix[:,1].astype(np.int64))
        L=np.searchsorted(pts_sorted, lin, side='left')
        R=np.searchsorted(pts_sorted, lin, side='right')
        xs,ys,zs,idxs=[],[],[],[]
        for l,r in zip(L,R):
            if r>l:
                xs.append(x_sorted[l:r]); ys.append(y_sorted[l:r]); zs.append(z_sorted[l:r])
                idxs.append(kept_idx_ordered[l:r])
        if not xs: continue
        bx=np.concatenate(xs); by=np.concatenate(ys); bz=np.concatenate(zs); bidx=np.concatenate(idxs)

        bcsv=os.path.join(csv_dir, f'{bid}_point_cloud.csv')
        ensure_exists(bcsv); pd.DataFrame({'X':bx,'Y':by,'Z':bz}).to_csv(bcsv,index=False)

        bl=laspy.LasData(header)
        bl.X=(bx-las.header.x_offset)/las.header.x_scale
        bl.Y=(by-las.header.y_offset)/las.header.y_scale
        bl.Z=(bz-las.header.z_offset)/las.header.z_scale
        for a in attrs:
            if hasattr(las,a): setattr(bl,a, getattr(las,a)[bidx])
        blas=os.path.join(las_dir, f'{bid}_point_cloud.las')
        ensure_exists(blas); bl.write(blas)

    log("Processing completed.")

if __name__=="__main__":
    t0=time.time(); main(); log(f"Total time: {time.time()-t0:.2f}s")
