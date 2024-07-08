# Cell Pose Reconstruction

## Usage

1. install related python libraries (e.g., pytorch, trackpy) following the instruction of CoTracker (see [README-COTRACKER.md](./README-COTRACKER.md)).
2. Include current workspace into your python env and go to the code dir.
```shell
export PYTHONPATH=`pwd`
cd custom
```
3. Arrage the data files into the following structure
```shell
$DATAROOT
├── Blebb
│   ├── Cell 12
│   ├── Cell 3
│   ├── Cell 5
│   ├── Cell 7
│   └── Cell 9
├── Cilio
│   ├── Cell 11
│   ├── Cell 17
│   ├── Cell 20
│   ├── Cell 5
│   └── Cell 8
├── DMSO
│   ├── Cell10
│   ├── Cell12
│   ├── Cell2
│   ├── Cell5
│   └── Cell8
└── YW
```
and for each *cell* dir, there exist
```shell
$DATAROOT/Blebb/Cell 12
├── After drug
│   ├── 12_Position 13 after _Crop001_RAW_ch00.pkl
│   ├── 12_Position 13 after _Crop001_RAW_ch00.tif
│   ├── MetaData
├── Before drug
│   ├── 12_Position 13 before_Crop001_RAW_ch00.pkl
│   ├── 12_Position 13 before_Crop001_RAW_ch00.tif
│   ├── 3d.pkl
│   ├── blender.pkl
│   ├── MetaData
│   ├── 
└── Cell to track.PNG
```
4. run codes to perform detection
```shell
# marker detetion
python3 detection.py --data_path $DATAROOT/DMSO/Cell2/After\ DMSO
```
5. (Optional) run segmentaiton
   After detection, you will get *images.mp4*. Use [sem-and-track](https://github.com/fishfishson/Segment-and-Track-Anything) to segment it. Put the segmentation results into dir *images_masks*:
   ```shell
   ├── After drug
   │   ├── images_masks
   |   |    ├── 0000.png
   |   |    ├── 0001.png
    ```
6. run tracking and pose estimation
```shell
# tracking
python3 tracking.py --data_path $DATAROOT/DMSO/Cell2/After\ DMSO --start_time 0 --duration 200 --vis_threshold 0.1 --mode full
# pose estimation
python3 pose_estimation.py --data_path $DATAROOT/DMSO/Cell2/After\ DMSO --radius_mode sem
```

PS: Many thanks for yizh4ng's codes from https://github.com/yizh4ng/3D-reconstruction-from-nuclei-rotation.custom/
