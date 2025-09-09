# LiDAR-Camera Extrinsic Calibration using Chessboard Target

This project provides a method for calibrating the extrinsic parameters between a LiDAR sensor and a camera using a chessboard calibration target. The calibration process aligns the coordinate systems of both sensors, enabling accurate sensor fusion for robotics, autonomous vehicles, and computer vision applications.

This project leverages the OpenCV and Open3D Python libraries for image processing and 3D point cloud manipulation.

This is part of a DeltaX recruitment, September 2025, in South Korea.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation

This codebase is developed and run on Python 3.12.0.

Clone the repository and install the dependencies using the following commands:
```bash
git clone https://github.com/miftahulumam/lidar_camera_calib_chessboard.git
cd lidar_camera_calib_chessboard
pip install -r requirements.txt
```
The folder and file structures in the directory is as follows:
```
lidar_camera_calib_chessboard
            ├── CalibData/
            |   ├── images/
            |   |     ├── image_107906765910.png
            |   |     ├── image_112606640240.png
            |   |     └── ....
            |   |   
            |   ├── pointclouds/
            |   |     ├── points_107906765910.pcd
            |   |     ├── points_112606640240.pcd               
            |   |     └── ....
            |   |
            |   ├── cropped_pcd/  # Created after running checkerboard_plane_seg_3d.py
            |   |     ├── points_107906765910.pcd
            |   |     ├── points_112606640240.pcd               
            |   |     └── ....            
            |   |
            |   └── checkerboard_pcd/  # Created after running checkerboard_plane_seg_3d.py
            |         ├── points_107906765910.pcd
            |         ├── points_112606640240.pcd               
            |         └── .... 
            |
            ├── results/
            |   ├── 3d_corners/  # Created after running checkerboard_corners_det_3d.py
            |   |     ├── all_cb_points_3d.npz
            |   |     ├── points_107906765910.pcd
            |   |     ├── points_112606640240.pcd               
            |   |     └── ....
            |   |
            |   ├── centroids/  # Created after running checkerboard_corners_det_3d.py
            |   |     ├── points_107906765910.pcd
            |   |     ├── points_112606640240.pcd               
            |   |     └── ....  
            |   |
            |   ├── corners/ # Created after running camera_intrinsic_calibration.py (save_fig = True)
            |   |     ├── image_107906765910.png_corners.png
            |   |     ├── image_112606640240.png_corners.png
            |   |     └── ....          
            |   |
            |   ├── intrinsics_and_cb_points/ # Created after running camera_intrinsic_calibration.py (save_fig = True)
            |   |     └── intrinsics_and_cb_points.npz
            |   |    
            |   └── cb_plane_equations.npz # Created after running checkerboard_plane_seg_3d.py
            |
            └── {all Python scripts}
            
```

## Usage
Run the script with following order:
* `camera_intrinsic_calibration.py` obtains chessboard points from camera images and perform intrinsic calibration.
* `checkerboard_plane_seg_3d.py` segments chessboard point cloud data.
* `checkerboard_corners_det_3d.py` detects/approximates chessboard corner points in point cloud data.
* `visualize_point_cloud.py` visualizes the result of segmentation and 3D corner approximation.
* `extrinsic_calibration_ransac.py` performs extrinsic calibration on segmented chessboard point cloud and extracted 2D corner points.
    * `extrinsic_calibration_selected_frame.py` performs extrinsic calibration using a single selected frames only (`points_275491298040.pcd` or `points_263292668860.pcd`)
* `visualize_projection.py` visualizes the LiDAR projection result after extrinsic calibration.
``

> Note: The repository pipeline will be updated to streamline the process.

## Contact

Miftahul Umam

Email:
miftahul.umam14@gmail.com