import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class pos2d:
    def __init__(self, *args):
        if len(args) == 1:
            self.x = args[0][0]
            self.y = args[0][1]
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]
    def __repr__(self):
        return 'pos2d[{}, {}, {}]'.format(self.x, self.y)
    def __str__(self):
        return 'pos2d[{}, {}]'.format(self.x, self.y)
    def np(self):
        return np.array([self.x, self.y])
    def np_homo(self):
        return np.array([self.x, self.y, 1])
    
class pos3d:
    def __init__(self, *args):
        if len(args) == 1:
            self.x = args[0][0]
            self.y = args[0][1]
            self.z = args[0][2]
        elif len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
    def __repr__(self):
        return 'pos3d[{}, {}, {}]'.format(self.x, self.y, self.z)
    def __str__(self):
        return 'pos3d[{}, {}, {}]'.format(self.x, self.y, self.z)
    def np(self):
        return np.array([self.x, self.y, self.z])
    def np_homo(self):
        return np.array([self.x, self.y, self.z, 1])

def calc_camera_pos(ex_m):    
    center = pos3d(np.linalg.inv(ex_m) @ np.array([0, 0, 0, 1]))
    look = pos3d(np.linalg.inv(ex_m) @ np.array([0, 0, -1, 1]))
    
    return center, look

def calc_reflection(v_in: pos3d, v_norm: pos3d):
    v_proj = v_norm.np() * np.dot(-v_in.np(), v_norm.np())
    v_out = v_in.np() + 2 * v_proj
    
    return pos3d(v_out)

def calc_sphere_center(in_m, ex_m, image_size: pos2d, pos_image: pos2d, radius_image, radius):
    [[f_x, _, c_x], [_, _, c_y], [_, _, _]] = in_m
    fov_x = math.atan(image_size.x / 2 / f_x)    
    pos_image = pos2d(pos_image.x - c_x, pos_image.y - c_y)
    
    theta = math.atan(radius_image / ((image_size.x / 2 / math.tan(fov_x)) + (2 * pos_image.x * pos_image.x * math.tan(fov_x) / image_size.x) + (2 * pos_image.x * radius_image * math.tan(fov_x) / image_size.x)))
    distance = radius / math.sin(theta)
    
    v = np.array([pos_image.x, pos_image.y, (image_size.x / 2) / math.tan(fov_x)])
    v = pos3d(v / np.linalg.norm(v) * distance)
    
    pos_sphere_center = np.linalg.inv(ex_m) @ v.np_homo()
    
    return pos3d(pos_sphere_center)

def calc_image2world(in_m, ex_m, image_size: pos2d, pos_image: pos2d, distance):
    [[f_x, _, c_x], [_, _, c_y], [_, _, _]] = in_m
    fov_x = math.atan(image_size.x / 2 / f_x)    
    pos_image = pos2d(pos_image.x - c_x, pos_image.y - c_y)
        
    v = np.array([pos_image.x, pos_image.y, (image_size.x / 2) / math.tan(fov_x)])
    v = pos3d(v / np.linalg.norm(v) * distance)
    
    pos_world = np.linalg.inv(ex_m) @ v.np_homo()
    
    return pos3d(pos_world)

def calc_image2world_vec(in_m, ex_m, image_size: pos2d, pos_image: pos2d):
    pos_point = calc_image2world(in_m, ex_m, image_size, pos_image, 1)
    pos_cam, _ = calc_camera_pos(ex_m)
    
    return pos3d(pos_point.x - pos_cam.x, pos_point.y - pos_cam.y, pos_point.z - pos_cam.z)

def calc_point_on_sphere(pos_ray_init: pos3d, vec_ray: pos3d, sphere_center: pos3d, sphere_radius):
    elem1 = vec_ray.x * vec_ray.x + vec_ray.y * vec_ray.y + vec_ray.z * vec_ray.z
    elem2 = 2 * (
        vec_ray.x * (pos_ray_init.x - sphere_center.x) +
        vec_ray.y * (pos_ray_init.y - sphere_center.y) +
        vec_ray.z * (pos_ray_init.z - sphere_center.z)
        )
    elem3 = (
        (pos_ray_init.x - sphere_center.x) * (pos_ray_init.x - sphere_center.x) + 
        (pos_ray_init.y - sphere_center.y) * (pos_ray_init.y - sphere_center.y) + 
        (pos_ray_init.z - sphere_center.z) * (pos_ray_init.z - sphere_center.z) -
        sphere_radius * sphere_radius
        )
    
    _, ray_distance = solve_quad_eq(elem1, elem2, elem3)
    point_on_sphere = pos3d(pos_ray_init.x + ray_distance * vec_ray.x,
                            pos_ray_init.y + ray_distance * vec_ray.y,
                            pos_ray_init.z + ray_distance * vec_ray.z
                            )
    return point_on_sphere

def calc_point_on_sphere2(pos_ray_init: pos3d, vec_ray: pos3d, sphere_center: pos3d, sphere_radius):
    elem1 = vec_ray.x * vec_ray.x + vec_ray.y * vec_ray.y + vec_ray.z * vec_ray.z
    elem2 = 2 * (
        vec_ray.x * (pos_ray_init.x - sphere_center.x) +
        vec_ray.y * (pos_ray_init.y - sphere_center.y) +
        vec_ray.z * (pos_ray_init.z - sphere_center.z)
        )
    elem3 = (
        (pos_ray_init.x - sphere_center.x) * (pos_ray_init.x - sphere_center.x) + 
        (pos_ray_init.y - sphere_center.y) * (pos_ray_init.y - sphere_center.y) + 
        (pos_ray_init.z - sphere_center.z) * (pos_ray_init.z - sphere_center.z) -
        sphere_radius * sphere_radius
        )
    
    ray_distance, _ = solve_quad_eq(elem1, elem2, elem3)
    point_on_sphere = pos3d(pos_ray_init.x + ray_distance * vec_ray.x,
                            pos_ray_init.y + ray_distance * vec_ray.y,
                            pos_ray_init.z + ray_distance * vec_ray.z
                            )
    return point_on_sphere

def calc_normal_on_sphere(point_on_sphere, sphere_center):
    normal = np.array([
        point_on_sphere.x - sphere_center.x, 
        point_on_sphere.y - sphere_center.y,
        point_on_sphere.z - sphere_center.z
        ])
    normal = pos3d(normal / np.linalg.norm(normal))
    return normal

def calc_stepper_ex_m(ex_m, distance):
    stepper_m = [
        [1, 0, 0, distance],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    ex_m_new = ex_m @ stepper_m
    return ex_m_new

def calc_ex_m(ex_m, ex_m_origin):
    ex_m_new = ex_m @ np.linalg.inv(ex_m_origin)
    
    return ex_m_new

def solve_quad_eq(a, b, c):
    """
    Solve ax^2 + bx + c = 0
    """
    answer_max = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    answer_min = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return answer_max, answer_min

def load_matrix(cam_idx, in_m_root, ex_m_root):
    """
    Load intrinsic matirx & extrinsic matrix
    """
    in_m_path = os.path.join(in_m_root, 'cam_{}'.format(cam_idx), 'intrinsic.txt')
    ex_m_path = os.path.join(ex_m_root, 'cam_{}'.format(cam_idx), 'extrinsic.txt')
    
    in_m = np.loadtxt(in_m_path)
    ex_m = np.loadtxt(ex_m_path)
            
    return in_m, ex_m

def get_specular_highlight(image):
    """
    Find position of specular highlight on image
    """
    _, _, _, max_pos = cv2.minMaxLoc(image)
    return pos2d(max_pos)

def get_circle(image):
    """
    Detect circle and return center point & radius
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=40, param2=15, minRadius=85, maxRadius=95)
    
    if circles is None:
        raise ValueError('no circle')
    
    x, y, r = circles[0][0]
    return pos2d(x, y), r

def write_circle(image, circle_center, circle_radius, color, save_path):
    cv2.circle(image, (int(circle_center.x), int(circle_center.y)), int(circle_radius), color * 255, 1)
    cv2.circle(image, (int(circle_center.x), int(circle_center.y)), 1, color * 255, 1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, image)
    
def write_circle_with_highlight(image, circle_center, circle_radius, highlight_center, color, save_path):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(image, (int(circle_center.x), int(circle_center.y)), int(circle_radius), color * 255, 1)
    cv2.circle(image, (int(circle_center.x), int(circle_center.y)), 1, color * 255, 1)
    
    cv2.circle(image, (int(highlight_center.x), int(highlight_center.y)), 1, (255, 0, 0), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, image)

def draw_camera_frustum(ax, in_m, ex_m, image_size: pos2d, color, size=10):
    [[f_x, _, _], [_, f_y, _], [_, _, _]] = in_m
    fov_x = 2 * math.atan(image_size.x / 2 / f_x)
    fov_y = 2 * math.atan(image_size.y / 2 / f_y)
    
    pos0 = pos3d(np.linalg.inv(ex_m) @ np.array([0, 0, 0, 1]))
    pos1 = pos3d(np.linalg.inv(ex_m) @ np.array([size * math.atan(fov_x / 2), size * math.atan(fov_y / 2), size, 1]))
    pos2 = pos3d(np.linalg.inv(ex_m) @ np.array([size * math.atan(fov_x / 2), -size * math.atan(fov_y / 2), size, 1]))
    pos3 = pos3d(np.linalg.inv(ex_m) @ np.array([-size * math.atan(fov_x / 2), -size * math.atan(fov_y / 2), size, 1]))
    pos4 = pos3d(np.linalg.inv(ex_m) @ np.array([-size * math.atan(fov_x / 2), size * math.atan(fov_y / 2), size, 1]))
    
    draw_line(ax, pos1, pos2, color=color)
    draw_line(ax, pos2, pos3, color=color)
    draw_line(ax, pos3, pos4, color=color)
    draw_line(ax, pos4, pos1, color=color)
    draw_line(ax, pos0, pos1, color=color)
    draw_line(ax, pos0, pos2, color=color)
    draw_line(ax, pos0, pos3, color=color)
    draw_line(ax, pos0, pos4, color=color)

def draw_sphere(ax, pos_center: pos3d, radius, color, alpha=0.2):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u) * np.sin(v) * radius + pos_center.x
    y_sphere = np.sin(u) * np.sin(v) * radius + pos_center.y
    z_sphere = np.cos(v) * radius + pos_center.z
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color=color, alpha=alpha)

def draw_point(ax, pos: pos3d, color, alpha=1):
    ax.scatter(pos.x, pos.y, pos.z, facecolor=color, alpha=alpha)

def draw_line(ax, pos1: pos3d, pos2: pos3d, color, alpha=1):
    """
    Draw line from pos1 to pos2
    """
    ax.plot([pos1.x, pos2.x], [pos1.y, pos2.y], [pos1.z, pos2.z], color=color, alpha=alpha)

def set_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    ax.set_xlim3d(-300, 300)
    ax.set_ylim3d(-300, 300)
    ax.set_zlim3d(0, 600)
    
    ax.view_init(elev=0, azim=-90, roll=0)
    
    return ax

if __name__ == '__main__':
    # cam_idx_pair = 10
    camera_parameter_path = '/root/workspace/DDIR/calibration'
    # in_m_root = 'C:/Users/owner/Documents/hpBRDF/stereo_calib/matrix_pair/origin_{}'.format(cam_idx_pair)
    # ex_m_root = 'C:/Users/owner/Documents/hpBRDF/stereo_calib/matrix_pair/origin_{}'.format(cam_idx_pair)
    sphere_image_root = '/bean/20240129_dataset_backup/240328_sphere'
    # sphere_highlight_root = '//bean.postech.ac.kr/data/yunseong/hpbrdf/data/light_position/metal_ball_WEAK_LIGHT'

    colors = plt.cm.jet(np.linspace(0, 1, 51))
    image_size = pos2d(512, 612)
    sphere_radius = 150 / 2
    # distance_light_source = 330.3

    angle_stage_sphere, gain_sphere, exposure_sphere = 0, 1.0, 500

    ###

    pos_stepper_list = [0, 50]
    cam_idx_list = [50]
    angle_stage_list = [80, 120, 160]

    ###

    ax = set_3d_plot()
    in_m, ex_m = load_matrix(0, camera_parameter_path, camera_parameter_path)


    color = colors[0]
    # handle with camera
    pos_cam, lookat_cam = calc_camera_pos(ex_m)
    draw_camera_frustum(ax, in_m, ex_m, image_size, color)

    sphere_image_path = os.path.join(sphere_image_root, '240328_sphere00/main/diffuse/wallpaper.png')

    # handle with sphere
    # path_image_sphere= os.path.join(sphere_image_path, '{}.png'.format(wavelength))
    image_sphere = cv2.imread(sphere_image_path)
    circle_center, circle_radius = get_circle(image_sphere)
    circle_save_path = './'
    os.makedirs(circle_save_path, exist_ok=True)
    write_circle(image_sphere, circle_center, circle_radius, color, os.path.join(circle_save_path, 'circle_0.png'))
    pos_sphere_center = calc_sphere_center(in_m, ex_m, image_size, circle_center, radius_image=circle_radius, radius=sphere_radius)

    draw_sphere(ax, pos_sphere_center, sphere_radius, color)

            
    plt.show()

    # # cam_idx_pair = 10
    # camera_parameter_path = '/root/workspace/DDIR/calibration/main'
    # # in_m_root = 'C:/Users/owner/Documents/hpBRDF/stereo_calib/matrix_pair/origin_{}'.format(cam_idx_pair)
    # # ex_m_root = 'C:/Users/owner/Documents/hpBRDF/stereo_calib/matrix_pair/origin_{}'.format(cam_idx_pair)
    # sphere_image_root = '/data1/20240129_dataset_backup/240328_sphere'
    # # sphere_highlight_root = '//bean.postech.ac.kr/data/yunseong/hpbrdf/data/light_position/metal_ball_WEAK_LIGHT'

    # colors = plt.cm.jet(np.linspace(0, 1, 51))
    # image_size = pos2d(512, 612)
    # sphere_radius = 150 / 2
    # # distance_light_source = 330.3
    
    # angle_stage_sphere, gain_sphere, exposure_sphere = 0, 1.0, 500
    
    # ###
    
    # pos_stepper_list = [0, 50]
    # cam_idx_list = [50]
    # angle_stage_list = [80, 120, 160]
    
    # ###

    # ax = set_3d_plot()
    # for pos_stepper in pos_stepper_list:
    #     sphere_image_path = os.path.join(sphere_image_root, str(angle_stage_sphere), str(pos_stepper), str(gain_sphere), str(exposure_sphere))
    #     for cam_idx in cam_idx_list:
    #         if cam_idx == cam_idx_pair: continue 
            
    #         # calculate basic information of camera
    #         wavelength = 450 + cam_idx * 4
    #         color = colors[cam_idx]
    #         in_m, ex_m = load_matrix(cam_idx, in_m_root, ex_m_root)
    #         # ex_m = calc_stepper_ex_m(ex_m, pos_stepper)
            
    #         # handle with camera
    #         pos_cam, lookat_cam = calc_camera_pos(ex_m)
    #         draw_camera_frustum(ax, in_m, ex_m, image_size, color)
            
    #         # handle with sphere
    #         path_image_sphere= os.path.join(sphere_image_path, '{}.png'.format(wavelength))
    #         image_sphere = cv2.imread(path_image_sphere)
    #         circle_center, circle_radius = get_circle(image_sphere)
    #         circle_save_path = 'C:/Users/owner/Documents/hpBRDF/figure/circle_detection'
    #         os.makedirs(circle_save_path, exist_ok=True)
    #         write_circle(image_sphere, circle_center, circle_radius, color, os.path.join(circle_save_path, 'circle_{}.png'.format(wavelength)))
    #         pos_sphere_center = calc_sphere_center(in_m, ex_m, image_size, circle_center, radius_image=circle_radius, radius=sphere_radius)
            
    #         draw_sphere(ax, pos_sphere_center, sphere_radius, color)
            
    #         # handle reflect
    #         for angle_stage in angle_stage_list:
    #             try:
    #                 print('[Stepper {}] [Cam {}] [{}nm] [Angle {}]'.format(pos_stepper, cam_idx, wavelength, angle_stage))
    #                 # find highlight point
    #                 gain_highlight, exposure_highlight = 1.0, 5
    #                 path_sphere_highlight = os.path.join(sphere_highlight_root, str(angle_stage), str(pos_stepper), str(gain_highlight), str(exposure_highlight), '{}.png'.format(wavelength))
    #                 gray_highlight = cv2.imread(path_sphere_highlight, cv2.IMREAD_GRAYSCALE)
    #                 pos_highlight2D = get_specular_highlight(gray_highlight)
    #                 vec_cam2highlight = calc_image2world_vec(in_m, ex_m, image_size, pos_highlight2D)
    #                 pos_highlight3D = calc_point_on_sphere(pos_cam, vec_cam2highlight, pos_sphere_center, sphere_radius)
    #                 draw_point(ax, pos_highlight3D, color, alpha=0.5)
    #                 draw_line(ax, pos_cam, pos_highlight3D, color, alpha=0.3)
                    
    #                 circle_save_path = 'C:/Users/owner/Documents/hpBRDF/figure/circle_detection'
    #                 os.makedirs(circle_save_path, exist_ok=True)
    #                 write_circle_with_highlight(image_sphere, circle_center, circle_radius, pos_highlight2D, color, os.path.join(circle_save_path, 'highlight_on_{}.png'.format(wavelength)))
    #                 write_circle_with_highlight(gray_highlight, circle_center, circle_radius, pos_highlight2D, color, os.path.join(circle_save_path, 'highlight_off_{}.png'.format(wavelength)))
                    
    #                 # find normal
    #                 normal_highlight = calc_normal_on_sphere(pos_highlight3D, pos_sphere_center)
    #                 draw_line(ax, pos_highlight3D, pos3d(pos_highlight3D.x + normal_highlight.x * 20, 
    #                                                     pos_highlight3D.y + normal_highlight.y * 20,
    #                                                     pos_highlight3D.z + normal_highlight.z * 20), color, alpha=0.5)
                    
    #                 # find light source
    #                 vec_highlight2source = calc_reflection(vec_cam2highlight, normal_highlight)
    #                 pos_source = calc_point_on_sphere2(pos_highlight3D, vec_highlight2source, pos_sphere_center, distance_light_source)
    #                 draw_line(ax, pos_highlight3D, pos_source, color, alpha=0.3)
    #                 draw_point(ax, pos_source, color, alpha=1)
    #             except Exception as e:
    #                 print(e)
    #                 continue
        
    # plt.show()
