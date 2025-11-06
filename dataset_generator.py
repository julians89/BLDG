import os, subprocess, math, numpy as np

BLENDER="blender"
BRIDGE="blender_bridge.py"
OUT_BASE="./out_dataset"
MESHES={
    #"box":"./meshes/box.ply",
    "a320":"./meshes/737_500.stl",
    "a320":"./meshes/a319_ceo.stl",
    "a320":"./meshes/a380.stl",
}

# Sampling grids
dists=[5,6,7,8,9,10,11]
yaws=np.deg2rad([0,45,90])
pitches=np.deg2rad([-10,0,10])

os.makedirs(OUT_BASE,exist_ok=True)

for cls,mesh in MESHES.items():
    for d in dists:
        for yaw in yaws:
            for pitch in pitches:
                out_dir=os.path.join(OUT_BASE,cls,f"d{d}_y{int(np.rad2deg(yaw))}_p{int(np.rad2deg(pitch))}")
                os.makedirs(out_dir,exist_ok=True)

                # place object at (0,0,d) with given pitch,yaw
                # camera coordinates: x --> left to right, y --> down to up, z --> near to far
                x,y,z=0,0,d
                roll=0

                cmd=[BLENDER, "-b","-P",BRIDGE,"--",mesh,out_dir,str(x),str(y),str(z),str(yaw),str(pitch),str(roll),"0"]
                
                print("Running"," ".join(cmd))
                subprocess.run(cmd,check=True)
