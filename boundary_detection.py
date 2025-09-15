import os
import glob
import random

import open3d as o3d
import numpy as np

def point_line_distance(p, a, b):
    """Shortest distance between point p and line segment ab (all 3D)."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # clamp to segment
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def approximate_plane_edges(pcd_path, tolerance=0.01, visualize=False):
    # Convert to Open3D PointCloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # --- Step 1: Convex Hull ---
    hull, _ = pcd.compute_convex_hull()

    # Hull vertices (3D coords)
    hull_vertices = np.asarray(hull.vertices)

    # Hull edges from triangle mesh
    triangles = np.asarray(hull.triangles)
    edges = set()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i+1) % 3]
            edges.add(tuple(sorted((a, b))))
    edges = list(edges)  # unique edges (pairs of vertex indices)

    # --- Find all boundary points from original point cloud ---
    eps = tolerance  # tolerance for "near edge"
    boundary_mask = np.zeros(len(points), dtype=bool)

    for (i, j) in edges:
        a, b = hull_vertices[i], hull_vertices[j]
        dists = np.array([point_line_distance(p, a, b) for p in points])
        boundary_mask |= dists < eps

    boundary_points = points[boundary_mask]

    # ---  Visualization ---
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color([0, 0, 1])   # red edges
    pcd.paint_uniform_color([0.8, 0.8, 0.8])       # blue points

    boundary_pcd = o3d.geometry.PointCloud()
    boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
    boundary_pcd.paint_uniform_color([1, 0, 0])  # green = boundary points

    if visualize:
        o3d.visualization.draw_geometries([boundary_pcd, pcd])

    return hull_ls, boundary_pcd

def distance_point_to_line(point, line_point, line_dir):
    """
    Compute the distance from a 3D point to a line defined by a point and a direction.
    """
    line_dir = line_dir / np.linalg.norm(line_dir)
    return np.linalg.norm(np.cross(point - line_point, line_dir))

def ransac_line_3d(points, threshold=0.05, max_iterations=1000):
    """
    Fit a 3D line using RANSAC.
    
    points: np.array of shape (N, 3)
    threshold: max distance to consider as inlier
    max_iterations: number of RANSAC iterations
    """
    best_inliers = []
    best_line = (None, None)

    n_points = points.shape[0]
    
    for _ in range(max_iterations):
        # Randomly pick 2 points
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx]
        line_dir = p2 - p1

        # Compute distances of all points to this line
        distances = np.array([distance_point_to_line(p, p1, line_dir) for p in points])
        
        # Find inliers
        inliers = points[distances < threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (p1, line_dir)
    
    # Refine line direction using PCA on inliers
    if len(best_inliers) > 0:
        centroid = np.mean(best_inliers, axis=0)
        cov = np.cov(best_inliers - centroid, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        line_dir = eigvecs[:, np.argmax(eigvals)]
        best_line = (centroid, line_dir)
    
    return best_line, best_inliers


if __name__ == "__main__":
# import npz file containing chessboard points and camera parameters
    root_dir = os.getcwd()
    data_file = os.path.join(root_dir, "results", "intrinsics_and_cb_points", "intrinsics_and_cb_points.npz")

    # Load camera intrinsic parameters
    data = np.load(data_file, allow_pickle=True)
    found_images = data['found_images']
    obj_points_list = data['obj_points_cam_list']
    img_points_list = data['img_points_list']
    K = data['K']
    distortion = data['distortion']

    # Load corresponding point cloud
    pcd_file_list = []

    for img_file in found_images:
        print(f"Processing image file: {img_file}")
        # Extract number from base filename (format: image_####.png)
        base_filename = os.path.basename(img_file)
        number_str = base_filename.split('_')[1].split('.')[0]
        # print(f"Extracted number string: {number_str}")

        pcd_file_list.append(os.path.join(root_dir, "CalibData", "checkerboard_pcd", f"points_{number_str}.pcd"))
        # print(f"Corresponding point cloud file: {pcd_file_list[-1]}")

    print("pcd_file_list: ", len(pcd_file_list))
    print("found_images: ", len(found_images))

    line_points = []
    line_directions = []

    for pcd_path in pcd_file_list:
        hull_ls, boundary_pcd = approximate_plane_edges(pcd_path)
        pcd = o3d.io.read_point_cloud(pcd_path).paint_uniform_color([0.8, 0.8, 0.8])
        # o3d.visualization.draw_geometries([boundary_pcd])

        # apply ransac to find a line equation
        line, inliers = ransac_line_3d(np.asarray(boundary_pcd.points), threshold=0.01, max_iterations=1000)

        print("Line point:", line[0])
        print("Line direction:", line[1])

        line_points.append(line[0])
        line_directions.append(line[1])
    
        # Visualize
        line_points_3d = np.array([line[0] + t * line[1] for t in np.linspace(-1, 1, 100)])
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(line_points_3d)
        line_pcd.paint_uniform_color([0, 0.4, 0])  # green line
        o3d.visualization.draw_geometries([line_pcd, boundary_pcd, pcd])    

    # visualize lines in 3D
    line_pcd = o3d.geometry.PointCloud()
    all_line_points = []
    for lp, ld in zip(line_points, line_directions):
        line_pts = np.array([lp + t * ld for t in np.linspace(-0.5, 0.5, 100)])
        all_line_points.append(line_pts)
        line_pcd_tmp = o3d.geometry.PointCloud()
        line_pcd_tmp.points = o3d.utility.Vector3dVector(line_pts)
        line_pcd_tmp.paint_uniform_color([0, 0.4, 0])
        line_pcd += line_pcd_tmp
    o3d.visualization.draw_geometries([line_pcd])   

    # classify lines into 2 groups based on direction
    line_points = np.array(line_points)
    line_directions = np.array(line_directions)

    line_group1 = []
    dir_group1 = []
    line_group2 = []
    dir_group2 = []

    for lp, ld in zip(line_points, line_directions):
        ld = ld / np.linalg.norm(ld)

        if abs(ld[2]) > 0.5:  # more vertical
            line_group1.append(lp)
            dir_group1.append(ld)
        else:
            line_group2.append(lp)
            dir_group2.append(ld)
    
    line_group1 = np.array(line_group1)
    dir_group1 = np.array(dir_group1)
    line_group2 = np.array(line_group2)
    dir_group2 = np.array(dir_group2)

    print(f"Group 1: {len(line_group1)} lines, Group 2: {len(line_group2)} lines")  

    # visualize groups
    line_pcd1 = o3d.geometry.PointCloud()
    all_line_points1 = []
    for lp, ld in zip(line_group1, dir_group1):
        line_pts = np.array([lp + t * ld for t in np.linspace(-0.5, 0.5, 100)])
        all_line_points1.append(line_pts)
        line_pcd_tmp = o3d.geometry.PointCloud()
        line_pcd_tmp.points = o3d.utility.Vector3dVector(line_pts)
        line_pcd_tmp.paint_uniform_color([0, 0.4, 0])
        line_pcd1 += line_pcd_tmp

    line_pcd2 = o3d.geometry.PointCloud()
    all_line_points2 = []
    for lp, ld in zip(line_group2, dir_group2):
        line_pts = np.array([lp + t * ld for t in np.linspace(-0.5, 0.5, 100)])
        all_line_points2.append(line_pts)
        line_pcd_tmp = o3d.geometry.PointCloud()
        line_pcd_tmp.points = o3d.utility.Vector3dVector(line_pts)
        line_pcd_tmp.paint_uniform_color([0.4, 0, 0])
        line_pcd2 += line_pcd_tmp
        
    # visualize checkerboard point clouds
    pcd_combined = o3d.geometry.PointCloud()
    for pcd_path in pcd_file_list:
        pcd = o3d.io.read_point_cloud(pcd_path).paint_uniform_color([0.8, 0.8, 0.8])
        pcd_combined += pcd
 

    o3d.visualization.draw_geometries([line_pcd1, line_pcd2, pcd_combined])
    
