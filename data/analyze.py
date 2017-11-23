import math
import numpy as np
import csv
import tf
import cv2
import pygame

# couple of colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

CSV_HEADER = ['x', 'y', 'z', 'yaw']

window_dimensions = (800, 600)
screen_dimensions = (window_dimensions[0] * 3, window_dimensions[1] * 3)

px = py = 0.0

fname = './churchlot_with_cars.csv'
waypoints = []
with open(fname) as wfile:
    reader = csv.DictReader(wfile, CSV_HEADER)
    for wp in reader:
        # move track to positive
        x = (float(wp['x']) + 5.0) * 50.0
        y = window_dimensions[1] - (float(wp['y']) * 50.0 - 1000.0)
        z = float(wp['z'])
        q = tf.transformations.quaternion_from_euler(0., 0., float(wp['yaw']))

        waypoints.append({'x': x, 'y': y, 'z': z, 'q': q})


def _init_screen():
    # produces error msgs and works without
    # pygame.init()
    # set caption of pygame window
    pygame.display.set_caption("Happy Robots Dashboard")
    # create screen wof window dimensions set in the debug.launch file
    screen = pygame.display.set_mode(window_dimensions, pygame.DOUBLEBUF)
    # set background color from debug.launch file
    screen.fill(BLACK)

    return screen

def _update_screen(screen, img):
    if screen is not None and img is not None:
        # if dashboard_img is set, resize it to window dimension, generate bytes and draw it with pygame
        image = pygame.image.fromstring(cv2.resize(img, window_dimensions).tobytes(), window_dimensions, 'RGB')
        # put on _screen
        screen.blit(image, (0, 0))
        # update pygame screen
        pygame.display.flip()

def getLocalXY(theta, x, y, px, py):
    # convert to local coordinates
    vx = x - px
    vy = y - py
    lx = vx*np.cos(theta) + vy*np.sin(theta)
    ly = -vx*np.sin(theta) + vy*np.cos(theta)
    return lx, ly

def getWorldXY(theta, lx, ly, px, py):
    # convert back to global coordinates
    x = lx*np.cos(theta) - ly*np.sin(theta) + px
    y = lx*np.sin(theta) + ly*np.cos(theta) + py

    return x, y

def _draw_track():
    global track_image, px, py
    # transform base waypoints to vertices for cv2.polylines
    xs = list()
    ys = list()
    for wp in waypoints:
        xs.append(wp['x'])
        #  normalize y values
        ys.append(wp['y'])
    vertices = [np.column_stack((xs, ys)).astype(np.int32)]

    px = xs[0]
    py = ys[0]

    # create empty image with screen dimensions
    track_image = np.empty((screen_dimensions[0], screen_dimensions[1], 3), dtype=np.uint8)
    # draw polylines of the track
    cv2.polylines(track_image, vertices, True, WHITE, 5)

def draw_orientation(img):
    global px, py
    for wp in waypoints:
        q = wp['q']
        angle = q[-1] * 180.0 / math.pi
        x = float(wp['x'])
        y = float(wp['y'])

        euler = tf.transformations.euler_from_quaternion([q[0], q[1], q[2], q[3]])
        theta = euler[2]

        lx, ly = getLocalXY(theta, x, y, px, py)

        p2x = round(lx + 40.0 * math.cos(theta))
        p2y = round(ly + 40.0 * math.sin(theta))

        p2x, p2y = getWorldXY(theta, p2x, p2y, px, py)

        p1x = int(x)
        p1y = int(y)

        p2x = int(p2x)
        p2y = int(p2y)

        cv2.line(img, (p1x, p1y), (p2x, p2y), RED, thickness=5)

track_image = None
_draw_track()
screen = _init_screen()
img = np.copy(track_image)
draw_orientation(img)
while (True):
    _update_screen(screen, img)