import os

import numpy as np
import open3d as o3d
import pandas as pd
import copy
import cv2
import itertools
from tqdm import tqdm
from open3d.open3d.geometry import voxel_down_sample, estimate_normals, create_rgbd_image_from_color_and_depth
from sklearn.neighbors import NearestNeighbors

width = 512
focal = float(width / (np.tan(np.pi / 4) * 2))  # for virtual camera, 1 pixel = 1 mm. width=512mm, focal = 256


def T3_matrix(alpha, beta, gamma, x, y, z):
    cos = np.cos(alpha)
    sin = np.sin(alpha)
    R_z = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    cos = np.cos(beta)
    sin = np.sin(beta)
    R_y = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    cos = np.cos(gamma)
    sin = np.sin(gamma)
    R_x = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    R = np.dot(np.dot(R_z, R_y), R_x)
    matrix = np.append(R, np.reshape([x, y, z], (3, 1)), axis=1)
    matrix = np.append(matrix, [[0, 0, 0, 1]], axis=0)
    return matrix


def display_inlier_outlier(cloud, ind):
    inlier_cloud = o3d.open3d.geometry.select_down_sample(cloud, ind)
    outlier_cloud = o3d.open3d.geometry.select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])   # yellow
    # target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.6f." % voxel_size)
    pcd_down = o3d.open3d.geometry.voxel_down_sample(pcd, voxel_size)
    pcd_down, ind = o3d.open3d.geometry.radius_outlier_removal(pcd_down, nb_points=30, radius=voxel_size*5)
    # display_inlier_outlier(pcd_down, ind)
    radius_normal = voxel_size * 2  # kdtree parameter
    # print(":: Estimate normal with search radius %.6f." % radius_normal)
    o3d.open3d.geometry.estimate_normals(pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.6f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, image_name=43):
    # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, width, focal, focal, width / 2.0, width / 2.0)
    # source_color = o3d.io.read_image("./save/RGB/{}.png".format(image_name))
    # source_depth = o3d.io.read_image("./save/depth/{}.png".format(image_name))
    # source_rgbd_image = o3d.open3d.geometry.create_rgbd_image_from_color_and_depth(
    #         source_color, source_depth, depth_scale=width / 2.0, convert_rgb_to_intensity=False)
    # source = o3d.geometry.create_point_cloud_from_rgbd_image(
    #         source_rgbd_image, pinhole_camera_intrinsic)
    # o3d.io.write_point_cloud("./save/pcd/{}.xyzrgb".format(image_name), source)
    source = o3d.io.read_point_cloud("./save/pcd/{}.xyzrgb".format(image_name))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    # draw_registration_result(source_down, target_down, np.identity(4))
    return source, source_down, source_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.0
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5  # 0.5
    # print(":: Apply fast global registration with distance threshold %.6f" \
    #         % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, transformation):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.6f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=50))
    return result


def rmv_ceiling(pcd, vis, threshold=-0.00005):
    ceiling = copy.deepcopy(pcd)
    ceiling_points = np.array(ceiling.points)
    ceiling_colors = np.array(ceiling.colors)
    # print(ceiling_points.max(axis=0), ceiling_points.min(axis=0))
    # print(ceiling_points.shape)
    mask = ceiling_points[:, 1] > threshold  # not x, y
    ceiling.points = o3d.utility.Vector3dVector(ceiling_points[mask])
    ceiling.colors = o3d.utility.Vector3dVector(ceiling_colors[mask])
    vis.create_window()
    vis.add_geometry(ceiling)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()


def img_to_3d(start=0, total_img=11, width=512):
    focal = float(width / (np.tan(np.pi / 4) * 2))  # for virtual camera, 1 pixel = 1 mm. width=512mm, focal = 256
    in_matrix = np.array([[focal, 0, 256], [0, focal, 256], [0, 0, 1]])  # intrinsic matrix as explain
    in_matrix_inv = np.linalg.inv(in_matrix)  # inverse intrinsic matrix
    progress = tqdm(total=total_img-start, desc="img_to_3d_progress")   # progress bar
    for image_name in range(start, total_img):
        front_rgb = "./save/RGB/{}.png".format(image_name)  # image name format for load image
        front_depth = "./save/depth/{}.png".format(image_name)
        output_name = "./save/pcd/result_output_{}.xyzrgb".format(image_name)  # save image to pointcloud as xyzrgb format
        outputFile = open(output_name, "w")  # create point cloud name
        img = cv2.imread(front_rgb, 1)  # read image as BGR sequence
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(str)
        # normalize image color as xyzrgb require
        img_depth = cv2.imread(front_depth, cv2.IMREAD_GRAYSCALE)  # read depth image
        for v, u in itertools.product(range(img.shape[0]), range(img.shape[1])):  # v and u inverse
            if img_depth[u][v] > 100:
                continue
            uv = np.array([u,v,1]) * img_depth[u][v]  # homogeneous array of uv using depth of image
            xyz = np.around(np.dot(in_matrix_inv, uv), decimals=5).astype(str)
            # around coordinate before save file
            # print("{} {} {} {} {} {} {} {}".format(
            #     xyz[0],  # x-value
            #     xyz[1],  # y-value
            #     xyz[2],  # z-value
            #     norm_img[u][v][2],  # r
            #     norm_img[u][v][1],  # g
            #     norm_img[u][v][0],  # b
            #     u,
            #     v,))
            outputFile.write("{} {} {} {} {} {}".format(  # save point cloud data as xyzrgb format
                xyz[0],  # x-value, divided by 1000 if need to convert to meter
                xyz[1],  # y-value
                xyz[2],  # z-value
                norm_img[u][v][2],  # r = cv2[2]
                norm_img[u][v][1],  # g = cv2[1]
                norm_img[u][v][0]  # b  = cv2[0]
                # normalize rgb value
            ) + "\n")
        progress.update(1)  # update progress bar


def o3d_img_to_3d(start=0, total_img=11, width=512):
    focal = float(width / (np.tan(np.pi / 4) * 2))  # for virtual camera, 1 pixel = 1 mm. width=512mm, focal = 256
    in_matrix = o3d.camera.PinholeCameraIntrinsic(width, width, focal, focal, 256, 256)  # intrinsic matrix as explain
    progress = tqdm(total=total_img-start, desc="o3d_img_to_3d")   # progress bar
    os.makedirs('./save/RGBD/', exist_ok=True)
    for image_name in range(start, total_img):
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, width, focal, focal, width / 2.0,
                                                                     width / 2.0)
        source_color = o3d.io.read_image("./save/RGB/{}.png".format(image_name))
        source_depth = o3d.io.read_image("./save/depth/{}.png".format(image_name))
        source_rgbd_image = o3d.open3d.geometry.create_rgbd_image_from_color_and_depth(
            source_color, source_depth, depth_scale=width / 2.0, convert_rgb_to_intensity=False)
        source = o3d.geometry.create_point_cloud_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        o3d.io.write_point_cloud("./save/pcd/{}.xyzrgb".format(image_name), source, True)
        # front_rgb = "./save/RGB/{}.png".format(image_name)  # image name format for load image
        # front_depth = "./save/depth/{}.png".format(image_name)
        # front_rgbd = "./save/RGBD/{}.png".format(image_name)
        # output_name = "./pcd/result_output_{}.xyzrgb".format(image_name)
        # img = o3d.io.read_image(front_rgb)  # read image as BGR sequence
        # img_depth = o3d.io.read_image(front_depth)  # read depth image
        # # normalize image color as xyzrgb require
        # img_rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(img, img_depth, depth_scale=256.0,
        #                                                                   depth_trunc=50.0)
        # pcd = o3d.geometry.create_point_cloud_from_rgbd_image(img_rgbd, in_matrix)
        # o3d.io.write_point_cloud(output_name, pcd, True)
        progress.update(1)  # update progress bar


def point_based_matching(points_trans, points_ref):
    # n = len(point_pairs)
    # print("point_pairs: ", point_pairs[0][0], point_pairs[0][1], type(point_pairs[0][0]))
    # print("points_trans: ", points_trans[0], type(points_trans[0]), np.array(points_trans).shape)
    # print("points_ref: ", points_ref[0], type(points_ref[0]), np.array(points_ref).shape)
    # points_trans = np.array(points_trans)
    # points_ref = np.array(points_ref)
    assert points_trans.shape == points_ref.shape  # check the pairs of points num equal or not
    n = points_trans.shape[0]  # n= num of pair points
    m = points_trans.shape[1]  # m= dimension of matrix
    # print("size: ", n, m)
    if n == 0:  # if no pair point return
        return None, None

    points_trans_mean = np.mean(points_trans, axis=0)  # calculate the center of source point cloud
    points_ref_mean = np.mean(points_ref, axis=0)  # calculate the center of target point cloud
    # print(points_trans[0], points_trans_mean)
    points_trans = np.subtract(points_trans, points_trans_mean)  # pre process for build H matrix
    points_ref = np.subtract(points_ref, points_ref_mean)
    # print(points_trans[0], points_trans.shape, points_ref.shape)
    points_mtx = np.dot(points_trans.T, points_ref)  # build H matrix
    # print(points_mtx)
    # x=u*e*v;  u: rotation , e: scaler, vT: rotation
    u, s, vT = np.linalg.svd(points_mtx)  # decomposition SVD matrix to U, S and VT
    # print("{}\n{}\n{}\n".format(u, s, vT))
    rot_mtx = np.dot(vT.T, u.T)  # calculate the rotation matrix

    if np.linalg.det(rot_mtx) < 0:  # check if rot_mtx is reflection matrix or not
        print ("sign")
        Vt[m-1, :] *= -1
        rot_mtx = Vt.T * U.T
        print('new R', rot_mtx)

    t_mtx = points_ref_mean.T - np.dot(rot_mtx, points_trans_mean.T)  # calculate the translation matrix
    # print("t_mtx: ", t_mtx.shape, points_ref_mean.T.shape, rot_mtx.shape, points_trans_mean.T.shape)
    T_mtx = np.identity(m + 1)  # initial transformation matrix
    T_mtx[:m, :m] = rot_mtx  # corresponding transformation matrix element replaced by rotation matrix element
    T_mtx[:m, m] = t_mtx  # corresponding transformation matrix element replaced by translation matrix element
    # print("T_mtx: ", T_mtx)
    return rot_mtx, t_mtx, T_mtx


def icp(points, reference_points, voxel_size, mtx_init=np.identity(4), max_iterations=50, tolerance=1e-3,
        point_pairs_threshold=10, verbose=False):

    points = np.array(points.points)  # get points from point cloud
    reference_points = np.array(reference_points.points)
    # print(points.shape, reference_points.shape)
    # assert points.shape == reference_points.shape
    n = points.shape[0]  # total number of points
    m = points.shape[1]  # matrix size
    src = np.ones((m + 1, n))  #
    trg = np.ones((m + 1, reference_points.shape[0]))
    src[:m,:] = copy.deepcopy(points.T)  # build source matrix
    trg[:m,:] = copy.deepcopy(reference_points.T)  # build target matrix
    nbrs = NearestNeighbors(n_neighbors=1).fit(trg[:m,:].T)  # initial target in Nearest Neighobrs
    prev_error = 0
    min_error = 1
    distance_threshold = voxel_size * 0.4  # set distance threshold
    T_mtx_final = mtx_init  # initialize Transformation matrix
    # print("src:", src.shape, trg.shape)
    # print(src[:, 0])

    progress = tqdm(total=max_iterations, desc="icp_implement", position=0, leave=True)
    for iter_num in range(max_iterations):
        found = 1
        points_trans = np.array([])
        points_ref = np.array([])

        distances, indices = nbrs.kneighbors(src[:m,:].T)
        # print(type(distances), distances.shape, type(indices), indices.shape)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:  # filter points if pair points < threshold
                # print("dis: ", distances[nn_index][0])
                # closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))
                # print(src[:, nn_index])
                points_trans = np.concatenate((points_trans, src[:, nn_index]))
                points_ref = np.concatenate((points_ref, trg[:, indices[nn_index][0]]))
        points_trans = points_trans.reshape(m+1, -1)  # reshape new pair points
        points_ref = points_ref.reshape(m+1, -1)
        # print(points_trans.shape, points_ref.shape, points_trans[:,0])
        # if only few point pairs, stop process
        # if verbose:
        #     print('number of pairs found:', points_trans.shape[1])
        if points_trans.shape[1] < point_pairs_threshold:    # if too few points break
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        r_mtx, t_mtx, T_mtx = point_based_matching(points_trans[:m, :].T, points_ref[:m, :].T)  # compute T_matrix
        # print(r_mtx)
        # print(t_mtx)
        # if r_mtx is not None:
        #     if verbose:
        #         print('Rotation:', r_mtx)
        #         print('Translation:', t_mtx)
        if r_mtx is None or t_mtx is None:
            if verbose:
                print('No better solution can be found!')
            break

        src = np.dot(T_mtx, src)  # transform source matrix for next loop
        T_mtx_final = np.dot(T_mtx_final, T_mtx)  # accumulate transformation matrix
        # mean_error = np.mean(np.square(distances))  # compute the error using two point distance
        error = np.mean(np.abs(np.subtract(points_ref, points_trans)))
        # error = np.abs(prev_error - mean_error)
        if error < tolerance:  # check convergence
            if verbose:
                # print("{} {} {}".format(points_ref[:,0], points_trans[:,0], error[:,0]))
                # print("error: ", prev_error, mean_error)
                print('{}: error {} below tolerance!'.format(iter_num, error))
            break
        # if verbose:
        #     print('\r------ iteration', iter_num, np.abs(prev_error - mean_error), '------')
        # prev_error = mean_error  # save error for next loop
        if error < min_error:  # save minimum error transform matrix
            T_mtx_min = copy.deepcopy(T_mtx_final)
            min_error = error
        if iter_num == max_iterations-1:  # mark not found if reach max iteration
            found = 0
        progress.set_description("icp_implement: error: %s, pairs: %s" % (error, points_trans.shape[1]))
        progress.update(1)
    if found == 0:  # take minimum error transform matrix
        T_mtx_final = T_mtx_min
        print("not found tolerance matrix", end=' ')
    # print("T_mtx_final: ", T_mtx_final)
    return T_mtx_final, src


def reconstruction(num_of_img_start, num_of_img, mode='o3d'):
    voxel_size = 0.00001
    trans_list = []
    vis = o3d.visualization.Visualizer()
    os.makedirs('./save/pcd/', exist_ok=True)
    target, target_down, target_fpfh = prepare_dataset(voxel_size, image_name=num_of_img_start)
    rmv_ceiling(target_down, vis)
    track_point = np.zeros((1, 3))
    track_arr = np.zeros((1, 3))  # initial track arr for record the camera movement
    lines = []  # connect track arr points by lines, record index later on
    track_gt_name = "./save/record.txt"  # name of track ground truth from record
    track_gt = pd.read_csv(track_gt_name, header=None, sep=' ')  # get the track ground truth from record
    track_gt_arr = np.array(track_gt.drop(track_gt.columns[3:], axis=1))  # take xyz only
    track_gt_arr = np.subtract(track_gt_arr, track_gt_arr[0]) / 2.6e3 * 1.1  # translate gt to global coordinate
    ex_matrix = T3_matrix(0, 0, np.pi, 0, 0, 0)[:3, :3]  # rotate z 90deg, y 60 deg to global coordinate
    progress = tqdm(total=num_of_img - num_of_img_start)
    for img in range(num_of_img_start + 1, num_of_img + 1):
        source, source_down, source_fpfh = prepare_dataset(voxel_size, image_name=img)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        print("\n", result_ransac)
        if mode == 'o3d':
            result = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size,
                                         result_ransac.transformation)
            trans_list.append(result.transformation)
        else:
            T_mtx, src = icp(source_down, target_down, voxel_size=voxel_size, mtx_init=result_ransac.transformation,
                             tolerance=1e-7, verbose=False)
            trans_list.append(T_mtx)
        source_temp = copy.deepcopy(source_down)
        track_point_pc = o3d.geometry.PointCloud()
        track_point_pc.points = o3d.utility.Vector3dVector(track_point)
        track_gt_arr[img] = np.dot(ex_matrix, track_gt_arr[img].T).T
        for i in range(1, len(trans_list) + 1):
            source_temp.transform(trans_list[-i])
            track_point_pc.transform(trans_list[-i])
        rmv_ceiling(source_temp, vis)
        source_temp_arr = np.array(source_temp.points)
        target_down_arr = np.array(target_down.points)
        if target_down_arr.shape[0] > source_temp_arr.shape[0]:
            size = source_temp_arr.shape[0]
        else:
            size = target_down_arr.shape[0]
        # error = np.mean(np.abs(np.subtract(target_down_arr[:size], source_temp_arr[:size])))
        track_point_T = np.array(track_point_pc.points)
        track_arr = np.concatenate((track_arr, track_point_T))
        lines.append([img - 1, img])
        target, target_down, target_fpfh = copy.deepcopy(source), copy.deepcopy(source_down), copy.deepcopy(
            source_fpfh)
        # progress.set_description("%s %s %s" % (target_down_arr[0], source_temp_arr[0],
        #                          np.subtract(target_down_arr[0], source_temp_arr[0])))
        progress.update(1)
    track_arr = track_arr.reshape(-1, 3)
    track = o3d.geometry.PointCloud()
    track.points = o3d.utility.Vector3dVector(track_arr)
    track.paint_uniform_color([1, 0.706, 0])  # yellow
    rmv_ceiling(track, vis)
    track_line = o3d.geometry.LineSet()
    track_line.points = o3d.utility.Vector3dVector(track.points)
    track_line.lines = o3d.utility.Vector2iVector(lines)
    track_line.paint_uniform_color([1, 0.706, 0])  # yellow
    vis.add_geometry(track_line)

    track_gt = o3d.geometry.PointCloud()
    track_gt.points = o3d.utility.Vector3dVector(track_gt_arr)
    track_gt.paint_uniform_color([0, 0.651, 0.929])  # blue
    rmv_ceiling(track_gt, vis)
    track_gt_line = o3d.geometry.LineSet()
    track_gt_line.points = o3d.utility.Vector3dVector(track_gt.points)
    track_gt_line.lines = o3d.utility.Vector2iVector(lines)
    track_gt_line.paint_uniform_color([0, 0.651, 0.929])  # blue
    vis.add_geometry(track_gt_line)

    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    distance = np.abs(np.subtract(track_gt_arr[:track_arr.shape[0]], track_arr))
    # for i in range(10):  # L2 distance, track_arr.shape[0]
        #     print(distance[i])
    print("Trajectory Error: %s" % distance.mean())
    vis.run()
    # draw_registration_result(source_down, target_down, trans_init)


if __name__ == "__main__":
    num_of_img_start = 0
    num_of_img = len(os.listdir('./save/RGB/'))-1
    mode = 'o3d'  # only operate o3d function, otherwise implement my own function.
    if mode == 'o3d':
        o3d_img_to_3d(start=num_of_img_start, total_img=num_of_img+1)
    else:
        img_to_3d(start=num_of_img_start, total_img=num_of_img+1)
    reconstruction(num_of_img_start=num_of_img_start, num_of_img=num_of_img, mode=mode)