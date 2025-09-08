import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

show_fig = True
save_fig = True
save_data = True

# image list
root_dir = os.getcwd()
img_dir = os.path.join(root_dir, "CalibData", "images")
img_list = glob.glob(os.path.join(img_dir, "*.png"))

# pcd_dir = os.path.join(root_dir, "CalibData", "pointclouds")
# pcd_list = glob.glob(os.path.join(pcd_dir, "*.pcd"))

print("Number of images:", len(img_list))

# Specify the number of inner corners in rows and columns on the chessboard
NUM_ROWS = 6
NUM_COLS = 5
CB_DIM = (NUM_ROWS, NUM_COLS)

# Specify the physical size of the chessboard square (in meters)
CB_SQUARE_SIDE_LENGTH = 0.20

obj_points = np.zeros((NUM_ROWS*NUM_COLS,3), np.float32)
obj_points[:,:2] = np.mgrid[0:NUM_ROWS,0:NUM_COLS].T.reshape(-1,2)

# Scale by the side length of a single chessboard square
obj_points = obj_points * CB_SQUARE_SIDE_LENGTH

# # Loop through images
found_images = []
frame_number = []
obj_points_list = [] 
img_points_list = [] 

# Specify the termination criteria for finding chessboard
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS , 30, 0.001)

for i, fname in enumerate(img_list):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CB_DIM, None)

    if found:
        found_images.append(fname)
        obj_points_list.append(obj_points)
        corners_refined = cv2.cornerSubPix(gray, corners, 
                                           (11,11),(-1,-1), 
                                           criteria)
        
        img_points_list.append(corners_refined)
    
        print(f"Corners found in {i+1}-th image: {fname}")

        # Draw and display the corners
        if show_fig:
            img_corners = cv2.drawChessboardCorners(img, CB_DIM, corners_refined, found)
            cv2.imshow('img', img_corners)
            cv2.waitKey(500)
            
        if save_fig:
            if not os.path.exists(os.path.join(root_dir, "results", "corners")):
                os.makedirs(os.path.join(root_dir, "results", "corners"))

            save_path = os.path.join(root_dir, "results", "corners", os.path.basename(fname + "_corners.png"))    
            cv2.imwrite(save_path, img_corners)

cv2.destroyAllWindows() 


# Calibrate
rep_error, K, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, 
                                                            img_points_list, 
                                                            gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(K)

print("\nRe-projection error : \n")
print(rep_error)

print("\nDistortion coefficients : \n")
print(distortion.ravel()) 

# print("rvecs:\n", rvecs)
# print("tvecs:\n", tvecs)

# import scipy.io

# scipy.io.savemat('intrinsics.mat', 
#                  {'K': K, 'distortion': distortion})

# Obtain 3d chessboard positions in camera frame
obj_points_cam_list = []

for i in range(len(rvecs)):
    rvec = rvecs[i]
    tvec = tvecs[i]

    R, _ = cv2.Rodrigues(rvec)
    obj_points_cam = (R @ obj_points.T).T + tvec.T
    obj_points_cam_list.append(obj_points_cam)

# visualize 3D chessboard positions
if show_fig:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, obj_points_cam in enumerate(obj_points_cam_list):
        ax.scatter(obj_points_cam[:, 0], obj_points_cam[:, 1], obj_points_cam[:, 2], label=f'Image {i+1}')  

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Chessboard Corner Positions in Camera Frame')
    ax.legend()
    plt.show()

# Save parameters
if save_data:
    if not os.path.exists(os.path.join(root_dir, "results", "intrinsics_and_cb_points")):
        os.makedirs(os.path.join(root_dir, "results", "intrinsics_and_cb_points"))

    np.savez(os.path.join(root_dir, "results", "intrinsics_and_cb_points", "intrinsics_and_cb_points.npz"),
            found_images = found_images,
            obj_points_cam_list = obj_points_cam_list,
            img_points_list = img_points_list,
            K = K, 
            distortion = distortion,
            rep_error = rep_error)
