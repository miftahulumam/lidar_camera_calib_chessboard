import open3d as o3d
import numpy as np
import cv2
import os 
import glob
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from PIL import Image

voxel_size = 0.05  # voxel size for downsampling

# Load camera intrinsic parameters
data_file = os.path.join(os.getcwd(), "results", "intrinsics_and_cb_points", "intrinsics_and_cb_points.npz")
data = np.load(data_file, allow_pickle=True)
found_images = data['found_images']
obj_points_list = data['obj_points_cam_list']
img_points_list = data['img_points_list']
K = data['K']
distortion = data['distortion']

# print(distortion)
## from OpenCV calibration
K = np.array([[263.12606449,            0.,   356.87436166],
              [          0.,  261.72012469,   180.91535839],
              [          0.,             0.,            1.]])

# # using solvePnP ransac
# T = np.array([[ 0.06903875,  0.99758621,  0.00744334, -0.22258491],
#               [-0.05094884,  0.01097717, -0.99864093, -0.09452099],
#               [-0.99631213,  0.06856569,  0.05158372, -0.86435474],
#               [ 0.,          0.,          0.,          1.        ]])

# Refined using solvePnP from inliers of RANSAC
# T = np.array([[ 0.07587531,  0.99708945,  0.00745482, -0.19220686],
#               [-0.05479364,  0.01163446, -0.99842992, -0.11182244],
#               [-0.99561066,  0.0753477,   0.05551693, -0.85588741],
#               [ 0.,          0.,          0.,          1.        ]])

# using solvePnp ransac (sept 16 2025)
# T = np.array([[ 0.15221661,  0.98818441, -0.01793517,  0.09303361],
#               [ 0.00139778, -0.01836185, -0.99983043,  0.08957058],
#               [-0.98834617,  0.15216573, -0.00417624, -0.88274316],
#               [ 0.,          0.,          0.,          1.        ]])

# refined solvePnP from RANSAC inliers (sept 16 2025)
T = np.array( [[ 1.32624143e-01,  9.91045563e-01, -1.54766927e-02, -5.14097476e-04],
               [-2.36328363e-04, -1.55830072e-02, -9.99878550e-01,  9.72196999e-02],
               [-9.91166374e-01,  1.32611693e-01, -1.83247080e-03, -8.93477022e-01],
               [ 0.,              0.,              0.,              1.            ]])

# using single-frame solvePnP (selected frame == points_263292668860.pcd)
# T = np.array([[ 0.0437566,   0.99857653,  0.03050024, -0.28127931],
#               [-0.01069664,  0.03099601, -0.99946227,  0.02620558],
#               [-0.99898496,  0.04340682,  0.01203769, -0.7115826 ],
#               [ 0.,          0.,          0.,          1.        ]])

# using single-frame solvePnP (selected frame == points_275491298040.pcd)
# T = np.array([[ 0.1289227,  0.99155596,  0.01398985,  0.01276075],
#               [ 0.01492056, 0.01216639, -0.99981466,  0.08365482],
#               [-0.99154239, 0.12910754, -0.01322605, -0.8811914 ],
#               [ 0.,         0.,          0.,          1.        ]])

# using six selected frames and solvePnP
# T = np.array([[ 1.38136690e-01,  9.90406184e-01, -3.72089143e-03,  1.20920938e-01],
#               [-3.24633068e-02,  7.72859278e-04, -9.99472629e-01, -9.14860247e-03],
#               [-9.89880997e-01,  1.38184633e-01,  3.22586200e-02, -6.13918863e-01],
#               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

## from MATLAB calibration
# K = np.array([[ 329.6019,       0.,   335.9598],
#               [      0.0, 328.7889,   190.5651],
#               [      0.0,        0.,     1.0]])

# distortion = np.array([[ -0.0832, 0.0335,  0.00,  0.00,  0.00]])

# # Load camera extrinsic parameters (from MATLAB calibration)
# T = np.array([[0.0285,  0.9995, -0.0099, -0.1311],
#               [0.0028, -0.0100, -0.9999, -0.0068],
#               [-0.9996, 0.0285, -0.0031, -0.0618],
#               [0.0,     0.0,     0.0,     1.0    ]]) # sept 9

# Load corresponding point cloud
root_dir = os.getcwd() 
pcd_file_list = []

for img_file in found_images:
    print(f"Processing image file: {img_file}")
    # Extract number from base filename (format: image_####.png)
    base_filename = os.path.basename(img_file)
    number_str = base_filename.split('_')[1].split('.')[0]
    print(f"Extracted number string: {number_str}")

    pcd_file_list.append(os.path.join(root_dir, "CalibData", "pointclouds", f"points_{number_str}.pcd"))
    print(f"Corresponding point cloud file: {pcd_file_list[-1]}")

# # Load point cloud
# pcd_dir = os.path.join(root_dir, "CalibData", "pointclouds")
# pcd_list = glob.glob(os.path.join(pcd_dir, "*.pcd"))    

# # pcd_file = os.path.join(root_dir, "CalibData", "pointclouds", "points_275491298040.pcd")
# # pcd_file = os.path.join(root_dir, "CalibData", "pointclouds", "points_263292668860.pcd")
# pcd_file = random.choice(pcd_list) 
# print(f"Using point cloud file: {pcd_file}")

print("Getting point cloud projections...")

for i, pcd_file in tqdm(enumerate(pcd_file_list)):
    # Extract number from base filename (format: points_####.pcd)
    base_filename = os.path.basename(pcd_file)
    number_str = base_filename.split('_')[1].split('.')[0]
    # print(f"Extracted number string: {number_str}")

    # Load corresponding image
    img_dir = os.path.join(root_dir, "CalibData", "images")
    image_file = os.path.join(img_dir, f"image_{number_str}.png")
    # print(f"Using image file: {image_file}")

    # load 3D checkerboard corners point cloud
    cb_points_3d_file = os.path.join(root_dir, "results", "3d_corners", os.path.basename(pcd_file))

    # load checkerboard points
    cb_plane_file = os.path.join(root_dir, "CalibData", "checkerboard_pcd", os.path.basename(pcd_file))

    # Read point cloud file
    pcd = o3d.io.read_point_cloud(pcd_file)
    cb_points_3d = o3d.io.read_point_cloud(cb_points_3d_file)
    cb_plane = o3d.io.read_point_cloud(cb_plane_file)

    # Downsample point cloud
    if voxel_size is not None or voxel_size > 0 :
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        cb_points_3d = cb_points_3d.voxel_down_sample(voxel_size=voxel_size)
        cb_plane = cb_plane.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(pcd.points)
    cb_points_3d_np = np.asarray(cb_points_3d.points)
    cb_plane_np = np.asarray(cb_plane.points)

    # Transform point cloud to camera coordinate system
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.matmul(T, points_homogeneous.T).T[:, :3]

    cb_points_3d_homogeneous = np.hstack((cb_points_3d_np, np.ones((cb_points_3d_np.shape[0], 1))))
    cb_points_3d_cam = np.matmul(T, cb_points_3d_homogeneous.T).T[:, :3]

    cb_plane_homogeneous = np.hstack((cb_plane_np, np.ones((cb_plane_np.shape[0], 1))))
    cb_plane_cam = np.matmul(T, cb_plane_homogeneous.T).T[:, :3]    

    # apply distortion if available
    if distortion is not None:
        k1, k2, p1, p2, k3 = distortion[0][:5]

        # all points
        x = points_cam[:, 0] / points_cam[:, 2]
        y = points_cam[:, 1] / points_cam[:, 2]
        r2 = x**2 + y**2
        x_distorted = x * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_distorted = y * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + p1*(r2 + 2*y**2) + 2*p2*x*y
        points_cam[:, 0] = x_distorted * points_cam[:, 2]
        points_cam[:, 1] = y_distorted * points_cam[:, 2]

        # checkerboard approx corners
        x_cb = cb_points_3d_cam[:, 0] / cb_points_3d_cam[:, 2]
        y_cb = cb_points_3d_cam[:, 1] / cb_points_3d_cam[:, 2]
        r2_cb = x_cb**2 + y_cb**2
        x_cb_distorted = x_cb * (1 + k1*r2_cb + k2*r2_cb**2 + k3*r2_cb**3) + 2*p1*x_cb*y_cb + p2*(r2_cb + 2*x_cb**2)
        y_cb_distorted = y_cb * (1 + k1*r2_cb + k2*r2_cb**2 + k3*r2_cb**3) + p1*(r2_cb + 2*y_cb**2) + 2*p2*x_cb*y_cb
        cb_points_3d_cam[:, 0] = x_cb_distorted * cb_points_3d_cam[:, 2]
        cb_points_3d_cam[:, 1] = y_cb_distorted * cb_points_3d_cam[:, 2]

        # checkerboard plane points
        x_plane = cb_plane_cam[:, 0] / cb_plane_cam[:, 2]
        y_plane = cb_plane_cam[:, 1] / cb_plane_cam[:, 2]  
        r2_plane = x_plane**2 + y_plane**2
        x_plane_distorted = x_plane * (1 + k1*r2_plane + k2*r2_plane**2 + k3*r2_plane**3) + 2*p1*x_plane*y_plane + p2*(r2_plane + 2*x_plane**2)
        y_plane_distorted = y_plane * (1 + k1*r2_plane + k2*r2_plane**2 + k3*r2_plane**3) + p1*(r2_plane + 2*y_plane**2) + 2*p2*x_plane*y_plane
        cb_plane_cam[:, 0] = x_plane_distorted * cb_plane_cam[:, 2]
        cb_plane_cam[:, 1] = y_plane_distorted * cb_plane_cam[:, 2] 

    # filter points in front of the camera
    points_cam = points_cam[points_cam[:, 2] > 0]
    cb_points_3d_cam = cb_points_3d_cam[cb_points_3d_cam[:, 2] > 0]
    cb_plane_cam = cb_plane_cam[cb_plane_cam[:, 2] > 0]

    # Project points onto image plane
    points_2d_homogeneous = np.matmul(K, points_cam.T).T 
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]

    cb_points_2d_homogeneous = np.matmul(K, cb_points_3d_cam.T).T
    cb_points_2d = cb_points_2d_homogeneous[:, :2] / cb_points_2d_homogeneous[:, 2:3]

    cb_plane_2d_homogeneous = np.matmul(K, cb_plane_cam.T).T
    cb_plane_2d = cb_plane_2d_homogeneous[:, :2] / cb_plane_2d_homogeneous[:, 2:3]  

    # Load image
    image = cv2.imread(image_file)   
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Filter points that are within image bounds
    valid_indices = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
    points_2d = points_2d[valid_indices]           
    points_cam = points_cam[valid_indices]

    valid_cb_indices = (cb_points_2d[:, 0] >= 0) & (cb_points_2d[:, 0] < image_width) & \
                    (cb_points_2d[:, 1] >= 0) & (cb_points_2d[:, 1] < image_height)
    cb_points_2d = cb_points_2d[valid_cb_indices] 
    cb_points_3d_cam = cb_points_3d_cam[valid_cb_indices]

    valid_plane_indices = (cb_plane_2d[:, 0] >= 0) & (cb_plane_2d[:, 0] < image_width) & \
                        (cb_plane_2d[:, 1] >= 0) & (cb_plane_2d[:, 1] < image_height)  
    cb_plane_2d = cb_plane_2d[valid_plane_indices]
    cb_plane_cam = cb_plane_cam[valid_plane_indices]


    # color points based on depth
    cmap = 'viridis'

    depths = points_cam[:, 2]
    depth_min = np.min(depths)      
    depth_max = np.max(depths)
    depth_norm = (depths - depth_min) / (depth_max - depth_min)
    colors = plt.get_cmap(cmap)(depth_norm)[:, :3] * 255

    depth_cb = cb_points_3d_cam[:, 2]
    depth_cb_min = np.min(depth_cb)
    depth_cb_max = np.max(depth_cb)
    depth_cb_norm = (depth_cb - depth_cb_min) / (depth_cb_max - depth_cb_min)
    colors_cb = plt.get_cmap(cmap)(depth_cb_norm)[:, :3] * 255

    depth_plane = cb_plane_cam[:, 2]
    depth_plane_min = np.min(depth_plane)
    depth_plane_max = np.max(depth_plane)
    depth_plane_norm = (depth_plane - depth_plane_min) / (depth_plane_max - depth_plane_min)
    colors_plane = plt.get_cmap(cmap)(depth_plane_norm)[:, :3] * 255

    # Visualize projected points on image
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. All points
    axs[0].imshow(image_rgb)
    axs[0].scatter(points_2d[:, 0], points_2d[:, 1], c=colors / 255.0, s=1)
    axs[0].set_title("Projected All Points")
    axs[0].axis('off')

    # 2. Checkerboard corners
    axs[1].imshow(image_rgb)
    axs[1].scatter(cb_points_2d[:, 0], cb_points_2d[:, 1], c=colors_cb / 255.0, s=10, edgecolors='w')
    axs[1].set_title("Projected Checkerboard Corners")
    axs[1].axis('off')

    # 3. Checkerboard plane
    axs[2].imshow(image_rgb)
    axs[2].scatter(cb_plane_2d[:, 0], cb_plane_2d[:, 1], c=colors_plane / 255.0, s=1)
    axs[2].set_title("Projected Checkerboard Plane")
    axs[2].axis('off')

    # save figure
    if not os.path.exists(os.path.join(root_dir, "results", "calibration_results")):
        os.makedirs(os.path.join(root_dir, "results", "calibration_results"))

    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "results", "calibration_results", f"result_{number_str}.png"),
                dpi=300)

    plt.close()

vis_dir = os.path.join(root_dir, "results", "calibration_results")
vis_list = glob.glob(os.path.join(vis_dir, "*.png")) 
vis_file = random.choice(vis_list) 
print(f"Using result file: {vis_file}")

vis_img = Image.open(vis_file)
vis_img.show()








