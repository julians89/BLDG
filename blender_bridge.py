import bpy, sys, os, json, numpy as np, bmesh, math

# ---------------- CLI ----------------
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
if len(argv) < 9:
    raise SystemExit("Usage: blender -b -P blender_bridge.py -- <mesh_path> <out_dir> x y z yaw pitch roll dist_norm")

mesh_path, out_dir, x, y, z, yaw, pitch, roll, dist_norm = argv
x, y, z = float(x), float(y), float(z)
yaw, pitch, roll = float(yaw), float(pitch), float(roll)
out_dir = os.path.abspath(out_dir)
os.makedirs(out_dir, exist_ok=True)

W_FULL, H_FULL = 1024, 512
BAND_ROWS = (int(0.375*H_FULL), int(0.625*H_FULL))  # 45° band rows
NORM_FAR = 10.0

# -------------- utils --------------
def save_pgm(path, arr01):
    arr = np.clip(arr01, 0.0, 1.0)
    h, w = arr.shape
    arr8 = (arr * 255.0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr8.tobytes())

def load_ply_ascii(filepath):
    """
    Minimal ASCII PLY loader (verts + triangular faces).
    Returns (V: Nx3 float32, F: Mx3 int32)
    """
    with open(filepath, "r") as f:
        if f.readline().strip() != "ply":
            raise ValueError("Not a PLY file")
        n_verts = n_faces = None
        is_ascii = False
        # header
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in header")
            line = line.strip()
            if line.startswith("format ascii"):
                is_ascii = True
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith("element face"):
                n_faces = int(line.split()[-1])
            if line == "end_header":
                break
        if not is_ascii:
            raise ValueError("This loader only supports ASCII PLY. Convert your file to ASCII.")
        # verts
        V = []
        for _ in range(n_verts):
            parts = f.readline().strip().split()
            V.append([float(parts[0]), float(parts[1]), float(parts[2])])
        # faces
        F = []
        for _ in range(n_faces):
            parts = f.readline().strip().split()
            k = int(parts[0])
            if k == 3:
                F.append([int(parts[1]), int(parts[2]), int(parts[3])])
            else:
                # simple fan triangulation for ngons
                idx = list(map(int, parts[1:1+k]))
                for i in range(1, k-1):
                    F.append([idx[0], idx[i], idx[i+1]])
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)
        return V, F

import struct, numpy as np, bpy

def load_stl(path, assume_units="auto"):
    """
    Load STL (binary or ascii) -> (V, F) with shared vertices.
    assume_units: "auto" | "m" | "mm"
      - "auto": if bbox is huge (>1000), assume mm and scale to meters.
    """
    with open(path, "rb") as f:
        header = f.read(80)
        rest = f.read()
    try:
        # Try binary: next 4 bytes = uint32 tri count
        tri_count = struct.unpack("<I", rest[:4])[0]
        expected = 50 * tri_count  # 12*4 + 12*4 + 12*4 + 2
        if len(rest[4:]) >= expected:
            # Parse binary
            V_list = []
            F_list = []
            off = 4
            for i in range(tri_count):
                # normal (ignored)
                # 3 vertices
                n = struct.unpack("<3f", rest[off:off+12]); off += 12
                v0 = struct.unpack("<3f", rest[off:off+12]); off += 12
                v1 = struct.unpack("<3f", rest[off:off+12]); off += 12
                v2 = struct.unpack("<3f", rest[off:off+12]); off += 12
                V_list.extend([v0, v1, v2])
                F_list.append((3*i, 3*i+1, 3*i+2))
                off += 2  # attr byte count
            V = np.asarray(V_list, dtype=np.float32)
            F = np.asarray(F_list, dtype=np.int32)
        else:
            raise ValueError("Not binary STL")
    except Exception:
        # Fallback: ASCII
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            V_list = []
            F_list = []
            tri_idx = 0
            for line in f:
                if line.lstrip().startswith("vertex"):
                    parts = line.strip().split()
                    V_list.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(V_list) % 3 == 0:
                        i = len(V_list)//3 - 1
                        F_list.append((3*i, 3*i+1, 3*i+2))
            V = np.asarray(V_list, dtype=np.float32)
            F = np.asarray(F_list, dtype=np.int32)

    # Deduplicate vertices so Blender can share them
    if len(V) == 0:
        raise ValueError("Empty STL")
    uniq, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    F = inv[F]
    V = uniq

    # Units: guess mm if bbox large
    if assume_units == "mm" or (assume_units == "auto" and np.max(np.abs(V)) > 1000.0):
        V *= 0.001  # mm -> m

    return V, F

def create_mesh_object(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.validate(clean_customdata=True)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj

# -------------- scene setup --------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.use_compositing = True
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.film_transparent = True
scene.render.resolution_x = W_FULL
scene.render.resolution_y = H_FULL
scene.render.resolution_percentage = 100
scene.display_settings.display_device = 'sRGB'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'

# Camera (equirect) – rotate 180° around Y so +X is centered horizontally
cam = bpy.data.cameras.new("Cam"); cam.type='PANO'; cam.panorama_type='EQUIRECTANGULAR'
cam.clip_start = 0.05
cam.clip_end = max(2000.0, 1.5*max(1.0, (x*x+y*y+z*z)**0.5))
cam_obj = bpy.data.objects.new("CamObj", cam)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.rotation_euler = (0.0, math.pi, 0.0)

# -------------- mesh load (no addons) --------------
ext = os.path.splitext(mesh_path)[1].lower()
if mesh_path.upper() == "BOX":
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0))
    obj = bpy.context.active_object
elif ext == ".ply":
    # your existing ASCII PLY loader (or reuse the earlier one)
    V, F = load_ply_ascii(mesh_path)
    obj = create_mesh_object("Mesh", V, F)
elif ext == ".stl":
    V, F = load_stl(mesh_path, assume_units="auto")  # converts mm->m if needed
    obj = create_mesh_object("Mesh", V, F)
else:
    raise ValueError(f"Unsupported mesh format: {ext}")

# Depth material (CameraData → View Distance → Emission)
mat = bpy.data.materials.new("DepthMat"); mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
for n in list(nodes): nodes.remove(n)
out  = nodes.new("ShaderNodeOutputMaterial")
emis = nodes.new("ShaderNodeEmission"); emis.inputs['Strength'].default_value = 1.0
camd = nodes.new("ShaderNodeCameraData")
links.new(camd.outputs['View Distance'], emis.inputs['Color'])
links.new(emis.outputs['Emission'], out.inputs['Surface'])
obj.data.materials.clear(); obj.data.materials.append(mat)

# Pose: rotation_euler = (pitch, roll, yaw) in radians
obj.location = (x, y, z)
obj.rotation_euler = (pitch, roll, yaw)

# -------------- compositor: depth + alpha to files --------------
# -------------- compositor: depth + alpha to files --------------
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()

rl = nt.nodes.new("CompositorNodeRLayers")
rgb2bw = nt.nodes.new("CompositorNodeRGBToBW")
comp = nt.nodes.new("CompositorNodeComposite")

fout_d = nt.nodes.new("CompositorNodeOutputFile")
fout_d.base_path = out_dir
fout_d.file_slots[0].path = "depth_full_1024x512"
fout_d.format.file_format = 'OPEN_EXR'
fout_d.format.color_mode  = 'BW'
fout_d.format.color_depth = '32'

fout_a = nt.nodes.new("CompositorNodeOutputFile")
fout_a.base_path = out_dir
fout_a.file_slots[0].path = "alpha_full_1024x512"
fout_a.format.file_format = 'PNG'
fout_a.format.color_mode  = 'BW'
fout_a.format.color_depth = '8'

nt.links.new(rl.outputs['Image'], rgb2bw.inputs['Image'])
nt.links.new(rgb2bw.outputs['Val'], comp.inputs['Image'])
nt.links.new(rgb2bw.outputs['Val'], fout_d.inputs['Image'])
nt.links.new(rl.outputs['Alpha'], fout_a.inputs['Image'])
# -------------- render --------------
bpy.ops.render.render(write_still=True)

# -------------- read back → numpy, crop, save --------------
def _find(prefix, ext):
    p = os.path.join(out_dir, f"{prefix}0001.{ext}")
    if os.path.exists(p): return p
    for f in sorted(os.listdir(out_dir)):
        if f.startswith(prefix) and f.endswith(f".{ext}"):
            return os.path.join(out_dir, f)
    raise FileNotFoundError(prefix)

depth_path = _find("depth_full_1024x512", "exr")
alpha_path = _find("alpha_full_1024x512", "png")

img_d = bpy.data.images.load(depth_path, check_existing=False)
buf_d = np.array(img_d.pixels[:], dtype=np.float32).reshape(img_d.size[1], img_d.size[0], 4)
depth_full = np.flipud(buf_d[..., 0])
y0, y1 = BAND_ROWS
depth_band = depth_full[y0:y1, :].astype(np.float32)

img_a = bpy.data.images.load(alpha_path, check_existing=False)
buf_a = np.array(img_a.pixels[:], dtype=np.float32).reshape(img_a.size[1], img_a.size[0], 4)
mask_band = (np.flipud(buf_a[..., 0])[y0:y1, :] > 0.5).astype(np.uint8)

np.save(os.path.join(out_dir, "depth.npy"), depth_band)
np.save(os.path.join(out_dir, "mask.npy"),  mask_band)

valid = depth_band[mask_band > 0]
if valid.size:
    # pick a robust far distance for visualization only
    norm_far = float(np.percentile(valid, 95))  # e.g., 95th percentile distance
    norm_far = max(norm_far, 5.0)               # keep some minimum contrast
else:
    norm_far = 10.0

vis = 1.0 - (depth_band / max(1e-6, norm_far))
vis = np.clip(vis, 0.0, 1.0) * mask_band   # keep background 0
save_pgm(os.path.join(out_dir, "range.pgm"), vis)

# -------------- metadata --------------
meta = {
    "image_size_full": [W_FULL, H_FULL],
    "band_rows_full": [int(y0), int(y1)],
    "lidar_size": [1024, 128],
    "camera_note": "Equirect pano; cam rotated (0,pi,0) so +X is center column",
    "gt_pose": {
        "translation_m": [x, y, z],
        "rotation_euler_xyz_rad": [pitch, roll, yaw]
    }
}
with open(os.path.join(out_dir, "scene.json"), "w") as f:
    json.dump(meta, f, indent=2)
