from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import json
from scipy.io import loadmat
import argparse
import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import os
import random
import math
import time
import pandas as pd
from tqdm import tqdm
import cv2


def rmv_ceiling_ground(pcd, vis, ceil_th, gd_th):
    pcd_tmp = copy.deepcopy(pcd)
    points = np.array(pcd_tmp.points)
    colors = np.array(pcd_tmp.colors)

    mask = points[:, 1] < ceil_th  # remove points at ceiling
    points = points[mask]
    colors = colors[mask]
    mask = points[:, 1] > gd_th  # remove points at ground
    points = points[mask]
    colors = colors[mask]
    pcd_tmp.points = o3d.utility.Vector3dVector(points)
    pcd_tmp.colors = o3d.utility.Vector3dVector(colors)
    # vis.create_window()
    # vis.add_geometry(pcd_tmp)
    # vis.poll_events()
    # vis.update_renderer()
    return points, colors


def show_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # R = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 180 * rot, 0))  # rotate 8 degree to align axis
    # pcd.rotate(R, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    points, colors = rmv_ceiling_ground(pcd, vis, ceil_th, gd_th)  # -1e-3, 3.5e-2
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])

    # bounding_box = o3d.geometry.PointCloud()
    # bounding_box.points = o3d.utility.Vector3dVector(points)
    # bb = bounding_box.get_axis_aligned_bounding_box()  # use bounding box see if align
    # bb.color = (1, 0, 0)
    # vis.add_geometry(bb)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run()
    # o3d.visualization.draw_geometries([bb])
    return points, colors


def save_main_png(points, colors, fig_path, name):
    points_2d = np.delete(points, 1, axis=1)
    # print(points_2d.shape, np.max(points_2d, axis=0), np.min(points_2d, axis=0))
    # print(points_2d[:, 1])
    plt.figure()
    # plt.plot(points_2d)
    plt.scatter(points_2d[:, 1], points_2d[:, 0], c=colors, s=1)
    x_lim = plt.gca().get_xlim()  # get max and min value at x axis
    y_lim = plt.gca().get_ylim()  # get max and min value at y axis
    # print("xlim: {}, ylim: {}".format(x_lim, y_lim))

    plt.savefig('{}{}'.format(fig_path, name))
    # plt.show()
    plt.close()
    x_lim = [x_lim[0] * scale, x_lim[1] * scale]  # scale and format as list
    y_lim = [y_lim[0] * scale, y_lim[1] * scale]
    return x_lim, y_lim


def save_png(points, colors, fig_path, name, x_lim, y_lim):
    points *= scale
    plt.figure()
    plt.scatter(points[:, 1], points[:, 0], c=colors, s=1)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig('{}{}'.format(fig_path, name))
    plt.close()


def get_target(args, mode='num'):
    if mode == 'num':
        if args.target == '1' or args.target == 1 or args.target == 'rack':
            target = 1
        elif args.target == '2' or args.target == 2 or args.target == 'cushion':
            target = 2
        elif args.target == '3' or args.target == 3 or args.target == 'lamp':
            target = 3
        elif args.target == '4' or args.target == 4 or args.target == 'cooktop':
            target = 4
        else:  # args.target == '0' or args.target == 0 or args.target == 'refrigerator':
            target = 0
    else:
        if args.target == '1' or args.target == 1 or args.target == 'rack':
            target = 'rack'
        elif args.target == '2' or args.target == 2 or args.target == 'cushion':
            target = 'cushion'
        elif args.target == '3' or args.target == 3 or args.target == 'lamp':
            target = 'lamp'
        elif args.target == '4' or args.target == 4 or args.target == 'cooktop':
            target = 'cooktop'
        else:  # args.target == '0' or args.target == 0 or args.target == 'refrigerator':
            target = 'refrigerator'
    return target


def seperate_target_obstacle(points, colors, target, fig_path, x_lim, y_lim, listnum=0):
    # target_points_list = []
    # target_colors_list = []
    # obstacle_idx = []
    # for target_points in target:
        # print(colors*255)
        # Find target at point cloud
        # print("target: ", target_points)

    # Find goal points by color code
    colors_tmp = copy.deepcopy(colors)
    points_tmp = copy.deepcopy(points)
    points_tmp = np.delete(points_tmp, 1, axis=1)   # del addition dimension

    # Get target point
    target_points = target[listnum]
    arr = np.array(np.around(colors_tmp * 255.), dtype=int)
    mask = np.all(arr == target_points, axis=1)  # colors, arr
    idx = np.where(mask)
    points_target = points_tmp[idx]
    colors_target = colors_tmp[idx]  # / 255.
    # points_target = np.delete(points_target, 1, axis=1)  # del addition dimension

    # Get obstacle points
    o_idx = np.where(~mask)
    points_obstacle = points_tmp[o_idx] * scale
    # points_obstacle = np.delete(points_tmp[o_idx], 1, axis=1) * scale  # del addition dimension
    # print("after: ", points_obstacle)
    points_obstacle.T[[1, 0]] = points_obstacle.T[[0, 1]]  # x,z to z, x
    points_obstacle = np.insert(points_obstacle, 2, 0.5, axis=1) # insert markersize for later use
    colors_obstacle = colors_tmp[o_idx]

    # Filter outlier points
    mask = np.all(abs(np.mean(points_target, axis=0) - points_target) < 2e-2, axis=1)  # filter
    idx = np.where(mask)
    points_target = points_target[idx] * scale
    points_target.T[[1, 0]] = points_target.T[[0, 1]]  # x, z to z, x
    colors_target = colors_target[idx]
    # print("find point: ", points_target.shape)

    # plot png to check result
    save_png(points_target, colors_target, fig_path, 'target_{}'.format(target_points), x_lim, y_lim)
    save_png(points_obstacle, colors_obstacle, fig_path, 'obstacle_{}'.format(target_points), x_lim, y_lim)
    # print("return: ", points_obstacle)
    # obs_points = points[o_idx]
    # print("obs_points: ", obs_points[0], obs_points[1])
    # obs_points.T[[2, 0]] = obs_points.T[[0, 2]]  # z, x to x, z
    # obs_points = np.delete(obs_points, 1, axis=1) * scale  # denormalize
    # obs_points = np.insert(obs_points, 2, 0.5, axis=1)  # insert markersize for later use
    # obs_colors = colors[o_idx]
    return points_target, colors_target, points_obstacle, colors_obstacle


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        start_point_list.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)
        cv2.imwrite('image_point.png', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


def save_start_map(goal_list, x_lim, y_lim, obstacle_list, obs_colors):
    plt.figure(figsize=(5.12, 5.12))  # pixel size
    plt.clf()
    # plot map and goal points
    if len(goal_list) > 1:
        for i in range(len(goal_list)):
            plt.plot(goal_list[i][0], goal_list[i][1], "ok", ms=1)
    else:
        plt.plot(goal_list[0], goal_list[1], "ok")

    plt.scatter(np.array(obstacle_list)[:, 0], np.array(obstacle_list)[:, 1], c=obs_colors, s=1)

    # standardize figure
    plt.axis('equal')
    plt.xlim((np.around(x_lim[0]), np.around(x_lim[1])))
    plt.ylim((np.around(y_lim[0]), np.around(y_lim[1])))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_start_point():
    width = 480  # pixel = 480x480
    focal = width / (np.tan(np.pi / 4) * 2)
    in_matrix = np.array([[focal, 0, 190], [0, focal, 270], [0, 0, 1]])  # dx, dy = center point at RRT map
    in_matrix_inv = np.linalg.inv(in_matrix)

    sensor_height_bev = 1
    for i in start_point_list:
        i.append(1)
        uv = np.array(i) * sensor_height_bev
        XY = np.dot(in_matrix_inv, np.reshape(uv, (3, 1))) * 10 * scale_num  # 9
        # print("pixel: ", i)
        # print("XY: ", XY)
    XY = np.array([XY[0][0], -XY[1][0]])
    return XY


class RRT:
    # initialize
    def __init__(self,
                 goal_list,
                 obstacle_list,  # obstacle points list
                 x_rand_area,  # x sampling range
                 y_rand_area,  # y sampling range
                 expand_dis=2,  # step for build tree: 2.0
                 goal_sample_rate=30,  # sample percentage toward goal points
                 max_iter=200, epoch=4):  # iteration for build RRT

        self.start = None
        self.goal = None
        self.goal_list = list(goal_list)
        # print("goal_mean: ", np.mean(goal_list, axis=0))
        # print("goal_list: ", self.goal_list)
        self.x_min_rand = x_rand_area[0]
        self.x_max_rand = x_rand_area[1]
        self.y_min_rand = y_rand_area[0]
        self.y_max_rand = y_rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None
        self.progress = tqdm(total=self.max_iter / epoch)
        self.epoch = epoch
        self.distance = scale
        print("x random range: ", (np.around(self.x_min_rand), np.around(self.x_max_rand)))
        print("y random range: ", (np.around(self.y_min_rand), np.around(self.y_max_rand)))

    def rrt_planning(self, obs_colors, start, goal, animation=True):  # start=[x,y]
        # print("start: ", start)
        print("GOAL CENTER: ", goal)
        self.start = Node(start[0], start[1])  # set start = struct with x, y, cost, parent
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        loop = self.max_iter / self.epoch
        plt.figure(figsize=(5.12, 5.12))  # pixel size
        # start = self.get_start_point()
        # self.start = Node(start[0][0], start[0][1])
        # progress = tqdm(total=self.max_iter)
        for i in range(self.max_iter):
            # 1. Sample a point
            rnd = self.sample()

            # 2. Find the nearest built tree point connected to rnd point
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]

            # 3. Get new node by provide step and calculated angle
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)

            # 4. Check collision: see any collision for a new point
            no_collision = self.check_segment_collision(new_node.x, new_node.y, nearest_node.x, nearest_node.y)
            if no_collision:
                self.node_list.append(new_node)

                # plot each step
                if animation:
                    time.sleep(1)
                    self.draw_graph(obs_colors, new_node, path, i)

                # Check every new node near to goal point
                # print("is_near_goal: ", self.is_near_goal(new_node))
                if self.is_near_goal(new_node):  # if close enough to goal point return no collision
                    # print("collision: ", self.check_segment_collision(new_node.x, new_node.y,
                    #                                 self.goal.x, self.goal.y))
                    # if self.check_segment_collision(new_node.x, new_node.y,
                    #                                 self.goal.x, self.goal.y):
                    last_index = len(self.node_list) - 1
                    path = self.get_final_course(last_index)  # Get the overall RRT path
                    path_length = self.get_path_len(path)  # Calculate path length
                    print("current path length：{}, # of nodes: {}".format(path_length, len(path)))

                    path_arr = np.array(path)  # L[::-1]: reverse read arr
                    path_pf = pd.DataFrame(path_arr)
                    path_pf.to_csv('./rrt/path.txt', header=False, index=False)
                    # fp = open('./rrt/record.txt', 'w')
                    # print(path, file=fp)
                    # fp.close()

                    if animation:
                        self.draw_graph(obs_colors, new_node, path, i)
                    return path
            self.progress.update(1)
            if i % loop + 1 >= loop:  # Restart RRT if RRT failed
                self.progress.refresh()
                self.progress.reset()
                self.start = Node(start[0], start[1])
                self.goal = Node(goal[0], goal[1])
                self.node_list = [self.start]
                path = None

    def sample(self):
        """ Uniform sample point and provide certain probability to sample towrad goal """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.x_min_rand, self.x_max_rand),  # random x,
                   random.uniform(self.y_min_rand, self.y_max_rand)]  # random y
        else:
            rnd = [self.goal.x, self.goal.y]
        return rnd

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        """ Find the nearest node from build tree to new node"""
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2
                  for node in nodes]
        min_index = d_list.index(min(d_list))  # get the index of tree
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        """ Calculate new node """
        new_node = copy.deepcopy(nearest_node)

        new_node.x += self.expand_dis * math.cos(theta)  # calculate x and y
        new_node.y += self.expand_dis * math.sin(theta)

        new_node.cost += self.expand_dis
        new_node.parent = n_ind

        return new_node

    def check_segment_collision(self, x1, y1, x2, y2):
        """ Check collision """
        for (ox, oy, radius) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]),  # new node
                np.array([x2, y2]),  # the nearest node from tree
                np.array([ox, oy])  # the obstacle point
            )
            # print("dd: ", dd)
            if dd <= radius ** 2:  # if projection < threshold: collision exist, threshold=obstacle point size
                return False
        return True  # no collision

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        """ Calculate the shortest distance from a new line (tree node connected a new node) and obstacle point p """
        if np.array_equal(v, w):  # if v and w are the same point
            return (p - v).dot(p - v)

        l2 = (w - v).dot(w - v)  # vector wv^2
        t = max(0, min(1, (p - v).dot(w - v) / l2))  # with t*(w-v): (pv dot wv)/|wv||wv| = |pv||wv|cos(theta)
        projection = v + t * (w - v)  # obstacle projection vector on build tree: e.g.: pv dot cos(theta)
        # t = max(0, min(1, (p - v) / (w-v)))
        # projection = v + t
        return (p - projection).dot(p - projection)

    def draw_graph(self, obs_colors, rnd=None, path=None, step=0):

        plt.clf()  # clear figure for new plot

        # Draw a new node on fig
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, '^k')

        # Plot RRT path
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y],
                             '-g')

        # Plot start point and goal point
        plt.plot(self.start.x, self.start.y, "og")
        # plt.plot(self.goal.x, self.goal.y, "or")
        if len(self.goal_list) > 1:
            for i in range(len(self.goal_list)):
                plt.plot(self.goal_list[i][0], self.goal_list[i][1], "ok", ms=1)
        else:
            plt.plot(self.goal.x, self.goal.y, "ob")

        # Plot obstacle points
        # print("obstacle_list: ", np.array(self.obstacle_list).shape)
        plt.scatter(np.array(self.obstacle_list)[:, 0], np.array(self.obstacle_list)[:, 1], c=obs_colors, s=1)
        # for idx, (ox, oy, size) in enumerate(self.obstacle_list):
        #     # plt.plot(ox, oy, "ok", ms=1 * size)  # 30
        #     plt.scatter(ox, oy, color=obs_colors[idx], s=1)

        # Plot path
        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        # Plot setup
        # plt.axis([np.around(self.x_min_rand), np.around(self.x_max_rand), np.around(self.y_min_rand),
        #           np.around(self.y_max_rand)])
        plt.axis('equal')
        plt.xlim((np.around(self.x_min_rand), np.around(self.x_max_rand)))
        plt.ylim((np.around(self.y_min_rand), np.around(self.y_max_rand)))
        plt.grid(True)
        # plt.xlim((-2, 8))  # fig axis limit scale, xlim
        # plt.ylim((-6, 12))
        # plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # plt.savefig('pict.png', bbox_inches='tight', pad_inches=0)
        plt.tight_layout()
        plt.savefig('./rrt/rrt{}.png'.format(step), bbox_inches='tight', pad_inches=0)
        plt.pause(0.01)

    def is_near_goal(self, node):
        for goal in self.goal_list:
            d = self.line_cost(node, goal)
            # print("distance: ", d)
            if d < self.expand_dis * 1.5:
                return True
        return False

    # @staticmethod
    def line_cost(self, node1, goal):
        distance = math.sqrt((node1.x - goal[0]) ** 2 + (node1.y - goal[1]) ** 2)
        if distance < self.distance: self.distance = distance
        self.progress.set_description("dis: {:.2f}".format(distance))
        # self.progress.set_description("x1: {:.2f}, x2: {:.2f}, y1: {:.2f}, y2: {:.2f}, dis: {:.2f}".format(
        #     node1.x, goal[0], node1.y, goal[1], distance))
        # print("x1: {:.2f}, x2: {:.2f}, y1: {:.2f}, y2: {:.2f}, dis: {:.2f}".format(
        #      node1.x, goal[0], node1.y, goal[1], distance))
        return distance

    def get_final_course(self, last_index):
        """ Get the path from goal point, check parent"""
        min_path = 10
        min_path_idx = 0
        for idx, goal in enumerate(self.goal_list):
            d = self.line_cost(self.node_list[last_index], goal)
            if d < min_path:
                min_path = d
                min_path_idx = idx

        path = [[self.goal_list[min_path_idx][0], self.goal_list[min_path_idx][1]]]  # [self.goal.x, self.goal.y]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    @staticmethod
    def get_path_len(path):
        """ Calculate path length"""
        path_length = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
        return path_length


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def RRT_implementation(start_point, goal_points, x_rand_area, y_rand_area, obstacle_list, obs_colors):
    print('Start RRT planning!')
    show_animation = True

    rrt = RRT(goal_points, x_rand_area=x_rand_area, y_rand_area=y_rand_area, obstacle_list=obstacle_list, expand_dis=1,
              max_iter=2000, epoch=4)
    path = rrt.rrt_planning(obs_colors, start=start_point, goal=np.mean(goal_points, axis=0), animation=show_animation)
    print('RRT finished!')
    if show_animation and path:
        plt.savefig('./rrt/rrt.png')
        plt.show()
        plt.close()


def read_rrt(scale_num=2, rrt_path='./rrt/backup/all_pc_scale-2/5_-5', list_num=0, mode='record'):
    # rrt_path = './rrt/backup/all_pc_scale-2/5_-5'
    if mode =='record:':
        rrt_arr = np.array(pd.read_csv('{}/{}/path.txt'.format(rrt_path, list_num), header=None)[::-1])
    else:
        rrt_arr = np.array(pd.read_csv('{}/path.txt'.format(rrt_path), header=None)[::-1])
    rrt_arr.T[[1, 0]] = rrt_arr.T[[0, 1]]  # (z,x) -> (x,z)
    rrt_arr = rrt_arr / scale_num
    return rrt_arr


def load_scene_semantic_dict():
    with open('../dataset/apartment_0/apartment_0/habitat/info_semantic.json', 'r') as f:
        return json.load(f)


def init_file(name, list_num=0):
    os.makedirs('./save/', exist_ok=True)
    fp = open('./save/record%s.txt' % name, 'w')
    fp.close()
    os.makedirs('./save/RGB%s/' % name, exist_ok=True)
    os.makedirs('./save/depth%s/' % name, exist_ok=True)
    os.makedirs('./save/semantic/', exist_ok=True)
    os.makedirs('./save/RGB_bev/', exist_ok=True)
    os.makedirs('./save/depth_bev/', exist_ok=True)
    os.makedirs('./save/semantic_bev/', exist_ok=True)
    # action = "move_forward"
    navigateAndSee("move_forward", list_num=list_num)


def record_path():
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    x, y, z, rw, rx, ry, rz = sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], \
        sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, \
        sensor_state.rotation.z
    deg = np.around(np.arccos(rw) * 360 / np.pi)
    # print("camera pose: x y z rw rx ry rz deg")
    # print(x, y, z, rw, rx, ry, rz, deg)  # convert quaternion to degree
    fp = open('./save/record%s.txt' % name, 'a')
    print(x, y, z, rw, rx, ry, rz, deg, file=fp)
    fp.close()
    return np.array([x, z, rw, deg])


def estimate_turn(rrt_arr_target, rrt_arr_current, current_degree):
    diff_x = rrt_arr_target[0] - rrt_arr_current[0]
    diff_z = rrt_arr_target[1] - rrt_arr_current[1]
    target_rotate = np.around(np.degrees(np.arctan(diff_z / diff_x)))
    if diff_x > 0:
        target_degree = 270 - target_rotate
    else:
        target_degree = 90 - target_rotate

    rotate = current_degree - target_degree  # current degree - target degree
    # print("diff_x: {}, diif_z: {}, target_rotate: {}, target_degree: {}, sensor[3]: {}, rotate: {}".format(
    #     diff_x, diff_z, target_rotate, target_degree, current_degree, rotate))
    if rotate > 0:
        action = "turn_right"
        if rotate > 180:
            action = "turn_left"
            rotate = 360 - rotate
    else:
        action = "turn_left"
        rotate = abs(rotate)
        if rotate > 180:
            action = "turn_right"
            rotate = 360 - rotate
    return rotate, action, target_degree


def take_turn(rotate, action, count, list_num=0):
    rotate_temp = copy.deepcopy(rotate)
    while rotate_temp > habitat_rotation / 2:
        count = navigateAndSee(action, count, name, list_num=list_num)
        rotate_temp = abs(rotate_temp - habitat_rotation)
    # print("action: ", action)
    return count  # , rotate_temp


def take_forward(forward, action, count, list_num=0):
    forward_temp = copy.deepcopy(forward)
    while forward_temp > habitat_forward / 2:
        count = navigateAndSee(action, count, name, list_num=list_num)
        forward_temp = abs(forward_temp - habitat_forward)
        # print(forward_temp)
    # print("action: ", action)
    return count, forward_temp


def final_turn(current_degree, list_num=0):
    if list_num == '1' or list_num == 1:  # rack
        target_degree = 200
    elif list_num == '2' or list_num == 2:  # cushion
        target_degree = current_degree
    elif list_num == '3' or list_num == 3:  # lamp
        target_degree = current_degree
    elif list_num == '4' or list_num == 4:  # cooktop
        target_degree = 90
    else:  # refrigerator
        target_degree = current_degree
    rotate = current_degree - target_degree  # current degree - target degree
    print("target_degree: {}, current_degree: {}, rotate: {}".format(target_degree, current_degree, rotate))
    if rotate > 0:
        action = "turn_right"
        if rotate > 180:
            action = "turn_left"
            rotate = 360 - rotate
    else:
        action = "turn_left"
        rotate = abs(rotate)
        if rotate > 180:
            action = "turn_right"
            rotate = 360 - rotate
    return rotate, action, target_degree


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def save_color_observation(observation, frame_number, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    # color_img = transform_rgb_bgr(observation)
    color_img = Image.fromarray(observation)  # color_img, observation
    # save color images
    color_img.save(os.path.join(out_folder, scene + filename_from_frame_number(frame_number)))


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def save_depth_observation(observation, frame_number, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    depth_img = Image.fromarray(
        (observation / 10 * 255).astype(np.uint8), mode="L")
    # save depth images
    depth_img.save(os.path.join(out_folder, scene + filename_from_frame_number(frame_number)))


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def fix_semantic_observation(semantic_observation, scene_dict):
    # The labels of images collected by Habitat are instance ids
    # transfer instance to semantic
    instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
    # print("instance_id_to_semantic_label_id: ", instance_id_to_semantic_label_id.shape,
    #       type(instance_id_to_semantic_label_id), instance_id_to_semantic_label_id)

    # print("shape: ", instance_id_to_semantic_label_id.shape)
    semantic_img = instance_id_to_semantic_label_id[semantic_observation]
    # print("semantic_img: ", semantic_img.shape, type(semantic_img), semantic_img)
    # print(semantic_img.mean(), semantic_img.max(), semantic_img.min())
    return semantic_img


def save_semantic_observation(semantic_obs, frame_number, scene_dict, out_folder='./save/generate/semantic/'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    semantic = fix_semantic_observation(semantic_obs, scene_dict)
    # img_sem101 = visualize_result(semantic, scene + filename_from_frame_number(frame_number), './save/generate/semantic/')
    seg_color = colorEncode(semantic, load_colors)
    img_sem101 = cv2.cvtColor(np.asarray(seg_color), cv2.COLOR_RGB2BGR)
    # semantic_img, seg_color
    # # cv2.imshow("semantic_101", img_sem101)
    # # semantic_img = Image.new("RGB", (semantic.shape[1], semantic.shape[0]))
    # semantic_img = Image.new("L", (semantic.shape[1], semantic.shape[0]))  # L -> black and white
    # semantic_img.putdata(semantic.flatten())
    # # save semantic images
    # semantic_img.save(os.path.join(out_folder, scene + filename_from_frame_number(frame_number)))
    # # _last_semantic_frame = np.array(semantic_img)
    return img_sem101


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors=loadmat('data/color101.mat')['colors'], mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        # labelmap_rgb = np.add(labelmap_rgb, (labelmap == label)[:, :, np.newaxis] * \
        #        np.tile(colors[label], (labelmap.shape[0], labelmap.shape[1], 1)), out=labelmap_rgb, casting="unsafe")
        # np.newaxis: create new axis/dimension
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label],
                                (labelmap.shape[0], labelmap.shape[1], 1))  # copy color as labelmap.shape

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


# def visualize_result(seg, img_name, dir_result):
#     # segmentation
#     seg_color = colorEncode(seg, colors)
#     semantic_img = cv2.cvtColor(np.asarray(seg_color), cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join("%s" % dir_result, img_name), semantic_img)  # semantic_img, seg_color
#     # aggregate images and save
#     # im_vis = Image.fromarray(seg_color)
#     # im_vis = Image.fromarray(seg_color).convert("RGB")
#     # im_vis.save(os.path.join("%s" % dir_result, img_name))
#     # semantic_img = cv2.cvtColor(np.asarray(im_vis), cv2.COLOR_RGB2BGR)
#     # cv2.imwrite(os.path.join("%s" % dir_result, img_name), semantic_img)
#     return semantic_img


# def convert_semantic(dir_path, image_name, save_path):
#     image_name = scene + filename_from_frame_number(image_name)
#     img = Image.open(os.path.join(dir_path, image_name))
#     # print(img.mode)
#     seg = np.multiply(np.asarray(img), 101 / 255)
#     seg = np.subtract(seg, 1)
#     # seg = np.around(seg)
#     print(seg.shape, seg)
#     print(seg.mean(), seg.max(), seg.min())
#     sem_img = visualize_result(seg, image_name, save_path)
#     return sem_img


def get_mask(image, list_num=0):
    '''# 0: refrigerator, 1: rack, 2: cushion, 3: lamp, 4: cooktop
    # 0: [0, 163, 255], 1: [0, 0, 255], 2: [255, 9, 112], 3: [255, 163, 0], 4: [6, 184, 255]'''
    image_tmp = copy.deepcopy(image).reshape(-1, 3)
    goal_colors = [np.array([0, 163, 255]), np.array([0, 0, 255]), np.array([255, 9, 112]),
                   np.array([255, 163, 0]), np.array([6, 184, 255])]
    goal = goal_colors[list_num]
    # 0: refrigerator, 1: rack, 2: cushion, 3: lamp, 4: cooktop
    # 0: [0, 163, 255], 1: [0, 0, 255], 2: [255, 9, 112], 3: [255, 163, 0], 4: [6, 184, 255]
    # print("RGB: ", image_tmp[0], image_tmp.shape)
    # for i in image_tmp:
    #     if(i[0] > 200):
    #         print(i)
    mask = np.all(image_tmp == goal, axis=1)
    # print(len(mask))
    idx = np.where(mask)
    # print("idx: ", len(idx))
    return idx, list_num


def mask_RGB(image, idx, list_num):
    if len(idx[0]) > 1:
        arr_idx = np.array([])
        image_tmp = copy.deepcopy(image).reshape(-1, 3)

        if list_num == 1:
            for i, arr in enumerate(image_tmp[idx[0]]):
                arr = np.array([arr[0], arr[1], arr[2] + 128])
                arr_idx = np.concatenate((arr_idx, arr))
        elif list_num == 3:
            for i, arr in enumerate(image_tmp[idx[0]]):
                mean = np.mean(arr)
                if mean > 128 and idx[0][i] > 133120:  # y>260, 255 > mean > 128
                    arr = np.array([arr[0] - 128, arr[1] - 128, 255])
                arr_idx = np.concatenate((arr_idx, arr))
        else:
            for i in image_tmp[idx[0]]:
                i = np.array([i[0], i[1], 255])
                # print(i)
                arr_idx = np.concatenate((arr_idx, i))
        arr_idx = arr_idx.reshape(-1, 3)
        image_tmp[idx[0]] = arr_idx
        image_tmp = image_tmp.reshape(512, 512, 3)
        # print(arr_idx.shape)
        # print("pixel: ", image_tmp[idx[0][0]], image_tmp[idx[0][0]][2], image_tmp[idx[0]])
        # A2d = np.arange(12).reshape(2, 6)
        # A2d[..., :, 5] = 0
        # print("arr: ", A2d)
        # image_tmp[idx[0]] = np.array([0, 0, 255])
        # print(image_tmp[idx[0]])
        # print(type(A2d), type(image_tmp[idx[0]]), A2d.shape, image_tmp[idx[0]].shape)
        # a = np.zeros((4, 3))
        # a.swapaxes(0, 1)[2] = 3
        # print(a)
        return image_tmp
    else:
        return image


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=habitat_forward)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=habitat_rotation)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=habitat_rotation)
        )
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def filename_from_frame_number(frame_number):
    return f"{frame_number:05d}.png"


def navigateAndSee(action="", num=0, name="", list_num=0):
    if action in action_names:
        observations = sim.step(action)
        # print("action: ", action)
        img_rgb = transform_rgb_bgr(observations["color_sensor"])
        img_depth = transform_depth(observations["depth_sensor"])
        img_sem = transform_semantic(observations["semantic_sensor"])
        save_color_observation(observations["color_sensor"], num, out_folder='./save/generate/images/')
        img_sem101 = save_semantic_observation(observations["semantic_sensor"], num,
                                               scene_dict=load_scene_semantic_dict(),
                                               out_folder='./save/generate/annotations/')
        save_depth_observation(observations["depth_sensor"], num, out_folder='./save/generate/depth/')
        # img_sem101 = convert_semantic(dir_path='./save/generate/annotations/', image_name=num, save_path='./save/generate/semantic/')
        idx, goal_colors = get_mask(img_sem101, list_num=list_num)
        img_rgb_new = mask_RGB(img_rgb, idx, goal_colors)
        cv2.imshow("RGB", img_rgb_new)
        # cv2.imshow("RGB", img_rgb)
        # cv2.imshow("depth", img_depth)
        # cv2.imshow("semantic", img_sem)
        # cv2.imshow("semantic_101", img_sem101)
        cv2.imwrite("./save/RGB{}/{}.png".format(name, num), img_rgb)
        # cv2.imwrite("./save/depth{}/{}.png".format(name, num), img_depth)
        # cv2.imwrite("./save/semantic/{}.png".format(num), img_sem)
        # cv2.imwrite(os.path.join('./save/generate/semantic/', scene + filename_from_frame_number(num)), img_sem101)
        cv2.waitKey(1)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        # print("camera pose: x y z rw rx ry rz")
        # print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], sensor_state.rotation.w,
        #       sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        fp = open('./save/record%s.txt' % name, 'a')
        print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], sensor_state.rotation.w,
              sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z, file=fp)
        fp.close()
        num += 1
        return num


def parser_arg():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--target', default=3)  # 0: refrigerator, 1: rack, 2: cushion, 3: lamp, 4: cooktop
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pcd_path = './semantic_3d_pointcloud/'
    points_path = 'point.npy'
    colors_path = 'color01.npy'
    fig_path = './figure/'
    rrt_path = './rrt'
    image_name = './rrt/rrt_start.png'
    test_scene = "../dataset/apartment_0/apartment_0/habitat/mesh_semantic.ply"
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(rrt_path, exist_ok=True)

    # initial variable, parameter
    ceil_th = -1e-3  # remove ceiling: -1e-3,
    gd_th = -3.5e-2  # remove ground: -3.5e-2
    scale_num = 2  # scale between RRT and scene
    scale = 10000 / 255 * scale_num
    rot = 0  # rotation for point cloud
    count = 0  # image sequence name
    start_point_list = []  # list for store click points
    args = parser_arg()  # for input target by command

    list_num = get_target(args)  # 0: refrigerator, 1: rack, 2: cushion, 3: lamp, 4: cooktop

    goal_colors = [[255, 0, 0], [0, 255, 133], [255, 9, 92], [160, 150, 20], [7, 255, 224]]

    load_colors = loadmat('data/color101.mat')['colors']
    scene = "apartment_0"
    habitat_rotation = 1.0  # rotation step
    habitat_forward = 0.01  # forward step
    name = ""  # RGB images folder suffix
    current_degree = 0  # initial direction

    # target = target_points[list_num]
    points = np.load('%s%s' % (pcd_path, points_path))  # 10000 / 255
    colors = np.load('%s%s' % (pcd_path, colors_path))  # / 255.

    points, colors = show_pcd(points, colors)
    # print('points max: {}, min: {}'.format(np.max(points, axis=0), np.min(points, axis=0)))
    # print('color max: {}, min: {}'.format(np.max(colors, axis=0), np.min(colors, axis=0)))

    x_lim, y_lim = save_main_png(points, colors, fig_path, '1st_floor')

    # print(np.array(target_colors).shape)
    # get list of target and obstacle
    goal_points, goal_colors_list, obs_points, obs_colors = \
        seperate_target_obstacle(points, colors, goal_colors, fig_path, x_lim, y_lim, listnum=list_num)
    # goal_points = goal_points_list
    # print("goal_points_list: {}, mean: {}, min: {}, max: {}".format(
    #     goal_points_list[0], np.mean(goal_points_list, axis=0),
    #     np.min(goal_points_list, axis=0), np.max(goal_points_list, axis=0)))
    # print("start_point: ", start_point)
    # print("goal_points: {}, mean: {}, min: {}, max: {}".format(
    #     goal_points_list[list_num][0], np.mean(goal_points_list[list_num], axis=0),
    #     np.min(goal_points_list[list_num], axis=0), np.max(goal_points_list[list_num], axis=0)))
    # goal_points = goal_points_list[list_num] * scale
    # print("goal_points: ", goal_points)
    # print(points[target_idx[list_num]])
    # obs_points = points[obs_idx[list_num]]
    # obs_colors = colors[obs_idx[list_num]]
    # obs_points = points[obs_idx]
    # # print("obs_points: ", obs_points[0], obs_points[1])
    # obs_points.T[[2, 0]] = obs_points.T[[0, 2]]
    # obs_points = np.delete(obs_points, 1, axis=1) * scale  # denormalize
    # obs_points = np.insert(obs_points, 2, 0.5, axis=1)  # insert markersize for later use
    # obs_colors = colors[obs_idx]
    # print("obs_points: {}, obs_colors: {}".format(obs_points.shape, obs_colors.shape))
    # print("obs: ", list(obs_points)[0], obs_points[0], x_lim, y_lim)

    save_start_map(list(goal_points), x_lim, y_lim, list(obs_points), obs_colors)  # show map for click start point
    img = cv2.imread(image_name, 1)  # top_rgb, front_rgb

    # click start point
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.imwrite('./save/color_img.jpg', img)
    cv2.destroyAllWindows()
    print("pixel: ", start_point_list)

    start_point = get_start_point()
    print("START POINT: ", start_point)
    RRT_implementation(start_point, goal_points, x_lim, y_lim, list(obs_points), obs_colors)

    # Read RRT path and initial start point
    rrt_arr = read_rrt(rrt_path=rrt_path, list_num=list_num, mode='Nrecord')  # mode='record'
    start_x = rrt_arr[0][0]  # agent in world space,  2.6, 2.5
    start_z = rrt_arr[0][1]  # 7.8, -2.5

    # Initialize simulator and agent
    init_file(name, list_num=list_num)
    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": -np.pi / 2,  # sensor pitch (x rotation in rads)
    }
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([start_x, 0.0, start_z])  # agent coordinate in world space
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("\n ==================================Start Navigation================================== \n")
    print("Discrete action space: ", action_names)

    sensor = record_path()  # get habitat sensor [x,z,rw,deg]
    print(sensor[0], rrt_arr[0][0], rrt_arr[-1][0], sensor[1], rrt_arr[0][1], rrt_arr[-1][1])
    print("sensor start point: ", sensor[0], sensor[1])
    print("rrt start point: {}, next: {}, total shape: {}".format(rrt_arr[0], rrt_arr[1], rrt_arr.shape))

    progress = tqdm(total=rrt_arr.shape[0] - 1)

    # Start Navigation
    for idx in range(1, rrt_arr.shape[0]):
        # print("target coords: {}, sensor coords: {}".format(rrt_arr[idx], rrt_arr[idx - 1]))
        # print("sensor: {}, {}".format(sensor[0], sensor[1]))
        rotate, action, current_degree = estimate_turn(rrt_arr[idx], rrt_arr[idx - 1], current_degree)
        # print("rotate: {}, action: {}".format(rotate, action))
        count = take_turn(rotate, action, count, list_num=list_num)  # rotate
        # print(rotate, action)
        move = np.array([rrt_arr[idx][0] - rrt_arr[idx - 1][0], rrt_arr[idx][1] - rrt_arr[idx - 1][1]])
        # print("move forward coord: ", move)
        move = np.sqrt(move[0] ** 2 + move[1] ** 2)
        # print("move forward: ", move)
        count, move = take_forward(move, "move_forward", count, list_num=list_num)  # move forward
        sensor = record_path()  # get current sensor info
        progress.update(1)
        progress.set_description("rotate {}: {:.2f}, forward: {:.2f}".format(action, rotate, move))
        # prev_rotate = rotate

    rotate, action, current_degree = final_turn(current_degree, list_num=list_num)
    count = take_turn(rotate, action, count, list_num=list_num)
    progress.update(1)
    print("Navigation Finished!")

    # control part
    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH = "f"
    print("#############################")
    print("use keyboard to control the agent")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for trun right  ")
    print(" f for finish and quit the program")
    print("#############################")

    while True:
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            count = navigateAndSee(action, count, name, list_num=list_num)
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            count = navigateAndSee(action, count, name, list_num=list_num)
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            count = navigateAndSee(action, count, name, list_num=list_num)
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
