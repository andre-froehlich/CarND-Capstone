import os, csv, re, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

try:
    import pygame
    from pygame.locals import *

    surfarray = pygame.surfarray
    if not surfarray: raise ImportError
except ImportError:
    raise ImportError, 'Error Importing Pygame/surfarray or Numeric'

pygame.init()
print 'Press the mouse button to advance image.'
print 'Press the "s" key to save the current image.'

window_dims = (800, 800)
screen_dims = (window_dims[1]*3, window_dims[0]*3)

clock = pygame.time.Clock()
clock.tick(1)

screen = pygame.display.set_mode(window_dims, 0, 32)


def init_screen(array_img, name):
    "displays a surface, waits for user to continue"
    surfarray.blit_array(screen, array_img)
    pygame.display.flip()
    pygame.display.set_caption(name)
    while 1:
        e = pygame.event.wait()
        if e.type == MOUSEBUTTONDOWN:
            break
        elif e.type == KEYDOWN and e.key == K_s:
            pygame.image.save(screen, name + '.bmp')
        elif e.type == QUIT:
            raise SystemExit


# couple of colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

base_path = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(base_path, '../ros/src/dashboard/original/final_waypoints.csv')
log_file_test = os.path.join(base_path, '../ros/src/dashboard/original/final_waypoints_test.csv')
churchlot_csv = os.path.join(base_path, 'churchlot_with_cars.csv')
base_waypoints = os.path.join(base_path, '../../fourth/base_waypoints.csv')

# pygame.display.set_caption("Visualize Track and Waypoints")
# screen = pygame.display.set_mode(window_dims, pygame.DOUBLEBUF)
# screen.fill(BLACK)

base4_wp = pd.read_csv(base_waypoints)
base4_xs = []
base4_ys = []
base_waypoints4 = dict()
for idx, item in base4_wp.iteritems():
    if "position" in idx:
        waypoint_idx = idx.split(".")[1]
        result = re.search("\d+", waypoint_idx)
        idx_key = int(result.group())
        if idx_key not in base_waypoints4:
            base_waypoints4[idx_key] = dict()

        base_waypoints4[idx_key][idx[-1]] = item.values[0]

for idx, wp in base_waypoints4.iteritems():
    base4_xs.append((float(wp['x']) + 10) * 25)
    base4_ys.append((float(wp['y']) + 5) * 25)

print "Length of basewaypoints: ", len(base4_xs)

final_waypoints = list()
with open(log_file, 'rb') as f:
    reader = csv.DictReader(f, ['id', 'pos_x', 'pos_y', 'pos_z', 'orient_x', 'orient_y', 'orient_z', 'orient_w'])
    for idx, wp in enumerate(reader):
        if idx == 0:
            continue
        final_waypoints.append(wp)

test_final_waypoints = list()
with open(log_file_test, 'rb') as f:
    reader = csv.DictReader(f, ['id', 'pos_x', 'pos_y', 'pos_z', 'orient_x', 'orient_y', 'orient_z', 'orient_w'])
    for idx, wp in enumerate(reader):
        if idx == 0:
            continue
        test_final_waypoints.append(wp)

curchlot_waypoints = list()
with open(churchlot_csv, 'rb') as f:
    reader = csv.DictReader(f, ['x', 'y', 'z', 'yaw'])
    for idx, wp in enumerate(reader):
        if idx == 0:
            continue
        curchlot_waypoints.append(wp)

xs = []
ys = []
ox, oy, oz, ow = [], [], [], []
for wp in final_waypoints:
    xs.append((float(wp['pos_x']) + 10) * 25)
    ys.append((float(wp['pos_y']) + 5) * 25)
    # ox.append(float(wp['orient_x']))
    # oy.append(float(wp['orient_y']))
    # oz.append(float(wp['orient_z']))
    # ow.append(float(wp['orient_w']))
#     if int(wp['id'].split('_')[0]) == 1:
#         print wp['id']
#         break

print "Length of final_waypoints_test: ", len(xs)

test_xs = []
test_ys = []
test_ox, test_oy, test_oz, test_ow = [], [], [], []
for wp in test_final_waypoints:
    test_xs.append((float(wp['pos_x']) + 10) * 25)
    test_ys.append((float(wp['pos_y']) + 5) * 25)
    # test_ox.append(float(wp['orient_x']))
    # test_oy.append(float(wp['orient_y']))
    # test_oz.append(float(wp['orient_z']))
    # test_ow.append(float(wp['orient_w']))
# if int(wp['id'].split('_')[0]) == 3:
#         print wp['id']
#         break

print "Length of final_waypoints_test: ", len(test_xs)

church_xs = []
church_ys = []
# test_ox, test_oy, test_oz, test_ow = [], [], [], []
for wp in curchlot_waypoints:
    church_xs.append((float(wp['x']) + 10) * 25)
    church_ys.append((float(wp['y']) + 5) * 25)

# while 1:
#     for i in range(20, len(xs)+20, 20):
#         track_image = np.zeros((screen_dims[0], screen_dims[1], 3), dtype=np.float32)
#
#         vertices = [np.column_stack((xs[i-20:i], ys[i-20:i])).astype(np.int32)]
#
#         vertices_test = [np.column_stack((test_xs[i-20:i], test_ys[i-20:i])).astype(np.int32)]
#
#         vertices_base = [np.column_stack((base4_xs, base4_ys)).astype(np.int32)]
#
#         track_image = cv2.polylines(track_image, vertices, False, BLUE, 2)
#         track_image = cv2.polylines(track_image, vertices_test, False, GREEN, 2)
#         track_image = cv2.polylines(track_image, vertices_base, False, WHITE, 5)
#
#         update(cv2.resize(track_image, window_dims), 'test')


# def update_screen(screen, img):
#     if screen is not None and img is not None:
#         # if dashboard_img is set, resize it to window dimension, generate bytes and draw it with pygame
#         image = pygame.image.fromstring(cv2.resize(img, window_dims).tobytes(), window_dims, 'RGB')
#         # put on _screen
#         screen.blit(image, (0, 0))
#         # update pygame screen
#         pygame.display.flip()
#
#
# while True:
#     update_screen(screen, track_image)
i = 20
while False:
    # if pygame.Clock.tick() >= 1000:
    print "{}th iteration".format(i // 20)
    track_image = np.zeros((screen_dims[0], screen_dims[1], 3), dtype=np.float32)

    vertices_base = [np.column_stack((base4_xs, base4_ys)).astype(np.int32)]
    track_image = cv2.polylines(track_image, vertices_base, False, WHITE, 2)

    # vertices = [np.column_stack((xs[i-20:i], ys[i-20:i])).astype(np.int32)]
    # vertices_test = [np.column_stack((test_xs[i-20:i], test_ys[i-20:i])).astype(np.int32)]
    # vertices_church = [np.column_stack((church_xs, church_ys)).astype(np.int32)]

    # track_image = cv2.polylines(track_image, vertices, False, BLUE, 2)
    # track_image = cv2.polylines(track_image, vertices_test, False, GREEN, 2)
    # track_image = cv2.polylines(track_image, vertices_church, True, YELLOW, 5)

    for x, y in zip(xs[i - 20:i], ys[i - 20:i]):
        track_image = cv2.circle(track_image, (int(x), int(y)), 5, BLUE, 3)

    for x, y in zip(test_xs[i - 20:i], test_ys[i - 20:i]):
        track_image = cv2.circle(track_image, (int(x), int(y)), 5, GREY, 3)

    for x, y in zip(church_xs, church_ys):
        track_image = cv2.circle(track_image, (int(x), int(y)), 2, YELLOW)

    i += 20
    surfarray.blit_array(screen, cv2.resize(track_image, window_dims))
    pygame.display.flip()
    pygame.display.set_caption("Visualize Waypoints")

    pygame.display.update()
    pygame.time.delay(750)

while False:
    track_image = np.zeros((screen_dims[0], screen_dims[1], 3), dtype=np.float32)
    xs = [408.12399999999997, 386.646, 364.266, 342.24800000000005, 322.89, 303.126, 280.61199999999997, 259.926, 242.86599999999999, 227.098, 213.346, 201.964, 196.652, 199.68, 209.406, 220.95600000000002, 235.646, 256.112, 275.03999999999996, 296.98199999999997, 319.084, 342.38599999999997, 366.58000000000004, 391.258, 416.54600000000005, 440.802, 464.62199999999996, 489.08000000000004, 513.34, 536.088, 559.288, 583.8340000000001, 604.4939999999999, 622.914, 639.7099999999999, 655.6219999999998, 670.154, 682.1659999999999, 695.256, 700.33, 699.6500000000001, 696.8520000000001, 691.77, 682.054, 666.626, 648.834, 630.98, 609.956, 589.1859999999999, 567.87, 546.304, 524.21, 502.19, 480.10800000000006, 459.832, 439.004, 416.03200000000004, 395.644, 376.42600000000004, 352.55, 333.468]
    ys = [2988.38, 2997.848, 3006.9539999999997, 3015.524, 3023.608, 3033.536, 3043.0060000000003, 3052.772, 3066.9719999999998, 3083.682, 3103.85, 3122.642, 3144.122, 3165.67, 3187.468, 3207.252, 3226.76, 3241.4120000000003, 3248.19, 3253.402, 3257.484, 3260.89, 3262.2039999999997, 3262.274, 3261.936, 3261.538, 3260.4880000000003, 3258.036, 3256.09, 3252.966, 3248.744, 3243.784, 3237.8540000000003, 3228.398, 3216.424, 3203.368, 3187.56, 3171.3720000000003, 3149.956, 3128.752, 3105.0460000000003, 3081.6639999999998, 3056.8, 3033.966, 3014.516, 3001.322, 2989.516, 2979.436, 2972.9700000000003, 2967.11, 2965.4939999999997, 2967.142, 2970.05, 2973.934, 2977.742, 2982.374, 2987.98, 2992.574, 2998.148, 3004.616, 3011.124]
    vertices_base = [np.column_stack((xs,ys)).astype(np.int32)]
    track_image = cv2.polylines(track_image, vertices_base, False, WHITE, 2)
    surfarray.blit_array(screen, cv2.resize(track_image, window_dims))
    pygame.display.flip()
    pygame.display.set_caption("Visualize Waypoints")

    pygame.display.update()
    pygame.time.delay(750)

    e = pygame.event.wait()
    if e.type == MOUSEBUTTONDOWN:
        break
    elif e.type == KEYDOWN and e.key == K_s:
        pygame.image.save(screen, name + '.bmp')
    elif e.type == QUIT:
        raise SystemExit
