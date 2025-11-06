# 🧩 Synthetic Depth Dataset Generator

This project generates **synthetic depth datasets** using **Blender’s Python API**.  
It renders depth and alpha maps from 3D meshes (e.g. `.stl`, `.ply`) across multiple camera poses and exports both raw and preprocessed data for machine learning or LiDAR simulation tasks.

---

## 📁 Project Structure
├── dataset_generator.py # Main script orchestrating dataset generation
├── blender_bridge.py # Blender rendering bridge
├── meshes/
│ └── box.ply # Example mesh
└── out_dataset/ # Generated dataset output
| └── box/
| | └── d5_y0_p0/
| | └── depth.npy
| | └── mask.npy
| | └── range.pgm
| | └── scene.json
| | └── depth_full_1024x512.exr
| | └── alpha_full_1024x512.png



## ⚙️ Requirements

- **Blender** ≥ 3.0 (must be callable via CLI: `blender`)
- **Python** ≥ 3.10
- **Dependencies**
  ```bash
  pip install numpy

## Usage

python dataset_generator.py

This will:

Create an output structure under ./out_dataset/

Invoke Blender in headless mode for each pose

Render depth and alpha maps at various (distance, yaw, pitch) configurations

Edit the configuration in dataset_generator.py to control mesh paths and sampling grids:

MESHES = {
    "box": "./meshes/box.ply",
    # "a320": "./meshes/a320.stl",
}

dists = [5, 6, 7, 8, 9, 10, 11]
yaws = np.deg2rad([0, 45, 90])
pitches = np.deg2rad([-10, 0, 10])

## Rendering Details

Camera type: Panoramic (Equirectangular)

Resolution: 1024×512

Vertical band used: 192–320 (≈ ±45° elevation)

Units: Meters
STL files are automatically scaled from millimeters if detected.

Render engine: Cycles, 1 sample, no denoising

Output includes both full-frame and cropped 45°-band data

## Example (Manual Run)

blender -b -P blender_bridge.py -- ./meshes/box.ply ./out_dataset/box/d5_y0_p0 0 0 5 0 0 0 0