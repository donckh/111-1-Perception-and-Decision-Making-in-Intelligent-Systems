import cv2
import numpy as np
import sympy as sp

points = []
sensor_height_bev = 2000  # in m
sensor_height = 1500


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


def object_to_image(f, z):
    in_mtx = np.array([[f / z, 0, 256], [0, f / z, 256], [0, 0, 1]])
    return in_mtx


class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.points = points
        self.height, self.width, self.channels = self.image.shape
        self.focal = self.width / (np.tan(np.pi / 4) * 2)
        self.in_matrix = np.array([[self.focal, 0, 256], [0, self.focal, 256], [0, 0, 1]])
        self.in_matrix_inv = np.linalg.inv(self.in_matrix)
        # focal length equation, vertical and horizontal are the same in this case. mm?
        # print(self.height, self.width, self.focal)

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        ### TODO ###
        # mid point = 256,256
        # convert image point to object point
        new_pixels = []
        for i in self.points:
            # X_bev = (i[0] - 256) * sensor_height_bev / self.focal
            # Y_bev = (i[1] - 256) * sensor_height_bev / self.focal  # Z
            # print("X_bev: ", X_bev, "Y_bev: ", Y_bev)
            i.append(1)
            uv = np.array(i) * sensor_height_bev
            print(i)
            XY = np.dot(self.in_matrix_inv, np.reshape(uv, (3, 1)))

            ex_matrix = T3_matrix(theta, phi, gamma, dx, sensor_height - sensor_height_bev, dz)
            # bev[x,y,z] = bev[z,y,x, tx,ty,tz]: rotate x axis -90deg, ty=-500
            print("transform: ", ex_matrix)
            new_point = np.dot(ex_matrix, np.reshape([[XY[0][0], XY[1][0], sensor_height_bev, 1]],
                                                     (4, 1)))  # object coordinate transform

            print("new_point[0][0]: ", new_point[0][0], "new_point[1][0]: ", new_point[1][0],
                  "new_point[2][0]: ", new_point[2][0], new_point)
            point = new_point[:3]
            print(self.in_matrix)
            print(point)
            uv = np.dot(self.in_matrix, point)
            uv = np.around(uv / uv[2][0], decimals=0).astype(int)
            # x = int(256 + round(self.focal * new_point[0][0] / new_point[2][0]))
            # y = int(256 + round(self.focal * new_point[1][0] / new_point[2][0]))
            # print("3D: ", x, y, type(x))
            new_pixels.append([uv[0][0], uv[1][0]])
        print(new_pixels)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append([x, y])
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


if __name__ == "__main__":
    pitch_ang = -np.pi / 2
    image_name = 69
    front_rgb = "./save/RGB/{}.png".format(image_name)  # to remove RGC on png later on
    top_rgb = "./save/RGB_bev/{}.png".format(image_name)

    # click the pixels on window
    img = cv2.imread(front_rgb, 1)  # top_rgb
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("out: ", points)
    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(gamma=pitch_ang, dy=sensor_height - sensor_height_bev)
    projection.show_image(new_pixels)
