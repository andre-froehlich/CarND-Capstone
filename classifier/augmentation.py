import cv2
import numpy as np
from scipy.stats import truncnorm

def get_trunc_norm(params):
    return truncnorm((params[0] - params[2]) / params[3],
                     (params[1] - params[2]) / params[3],
                     loc=params[2],
                     scale=params[3])


def augmentation_pipeline(img,
                          brightness=None,
                          color=None,
                          shadow=None,
                          blur=None,
                          shift=None,
                          rot=None,
                          perspective=None,
                          aug_prob=0.5,
                          flip_prob=0.5):

    # Check, if augment at all
    if np.random.random() > aug_prob:
        return img

    # Change brightness
    if brightness is not None:
        if brightness:
            img = aug_brightness(img)
        else:
            img = aug_brightness(img, brightness)

    # Augment color
    if color is not None:
        if color:
            img = aug_color(img)
        else:
            img = aug_color(img, color)

    # Add Shadow
    if shadow is not None:
        img = aug_shadow(img)

    # Blur
    if blur is not None:
        if blur:
            img = aug_blur(img)
        else:
            img = aug_blur(img, blur)

    # Shift
    if shift is not None:
        if shift:
            img = aug_shift(img)
        else:
            img = aug_shift(img, shift)

    # Rotate
    if rot is not None:
        if rot:
            img = aug_rotation(img)
        else:
            img = aug_rotation(img, rot)

    # Perspective transform
    if perspective is not None:
        if perspective:
            img = aug_perspective(img)
        else:
            img = aug_perspective(img, perspective)

    # Flip
    if np.random.random() <= flip_prob:
        img = np.fliplr(img)

    return img

def aug_brightness(img_in, params=(0.5, 1.5, 0.0, 1.0)):
    """
    Augment picture in brightness
    :param img_in: image in BGR (openCV standard)
    :return: image in BGR, augmented in brightness
    """
    # Change to HSV color space and transform to float64 array
    img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    img_out = np.array(img_out, dtype=np.float64)
    # Set a random number for brightness adjustment and adjust
    rand_bright = get_trunc_norm(params).rvs()

    img_out[:, :, 2] = img_out[:, :, 2] * rand_bright
    # set every value higher than 255 to 255
    img_out[:, :, 2][img_out[:, :, 2] > 255] = 255
    img_out[:, :, 2][img_out[:, :, 2] < 0] = 0
    # Transform back (unit8 and BRG)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
    return img_out

def aug_color(img_in, params=(-20.0, 20.0, 0.0, 10.0)):
    """
    Manipulates color channelwise by random (channel and value)
    :param img_in: BGR image (cv2 standard)
    :return: BGR image manipulated in color
    """
    # Convert to float64 array
    img_out = np.array(img_in, dtype=np.float64)

    distribution = get_trunc_norm(params)
    for i in range(3):
        rand_manipulation = round(distribution.rvs())
        # print(rand_manipulation)
        img_out[:, :, i] = img_out[:, :, i] + rand_manipulation
        img_out[:, :, i][img_out[:, :, i] > 255] = 255
        img_out[:, :, i][img_out[:, :, i] < 0] = 0

    # Convert back and return
    return np.array(img_out, dtype=np.uint8)

def aug_shadow(img_in):
    # Random values according to image size
    width = np.shape(img_in)[1]
    height = np.shape(img_in)[0]
    top_y = width * np.random.uniform()
    bot_y = width * np.random.uniform()
    top_x = 0
    bot_x = height

    # Color space and mask
    img_hls = cv2.cvtColor(img_in, cv2.COLOR_BGR2HLS)

    shadow_mask = 0 * img_hls[:, :, 1]
    x_m = np.mgrid[:height, :width][0]
    y_m = np.mgrid[:height, :width][1]
    shadow_mask[((x_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (y_m - top_y)) >= 0] = 1

    # Augment
    # Random bright .25 and .75
    # rand_bright = np.random.randint(low=8, high=7) / 100
    # rand_bright = 0.5 + np.random.uniform()
    rand_bright = (8 + (1.5 * np.random.uniform())) / 10

    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0
    if np.random.randint(2) == 1:
        img_hls[:, :, 1][cond1] = img_hls[:, :, 1][cond1] * rand_bright
    else:
        img_hls[:, :, 1][cond0] = img_hls[:, :, 1][cond0] * rand_bright

    img_hls[:, :, 1][img_hls[:, :, 1] > 255] = 255
    img_hls[:, :, 1][img_hls[:, :, 1] < 0] = 0

    img_hls = np.array(img_hls, dtype=np.uint8)

    # Transform back to BGR Color Space
    img_out = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)

    return img_out

def aug_blur(img_in, params=(1.0, 5.0, 1.0, 1.5)):
    """
    Blurs the image
    :param img_in: BGR image (cv2 standard)
    :return: BGR image with blur
    """
    # rand_kernel_size =  abs(int(np.random.normal(0.0, 10.0)))
    # rand_kernel_size = np.random.randint(low=5, high=21)
    rand_kernel_size = int(round(get_trunc_norm((params)).rvs()))
    rand_kernel = (rand_kernel_size, rand_kernel_size)
    return cv2.blur(img_in, rand_kernel)

def aug_shift(img_in, params=(-0.05, 0.05, 0.0, 0.02)):
    """
    Shifts picture in x and y by random (max. 15 % of pixels)
    :param img_in: BGR image (cv2 standard)
    :return: BGR image shifted
    """
    distribution = get_trunc_norm(params)

    shift_x = distribution.rvs()
    shift_y = distribution.rvs()

    # Pixels
    rand_shift_x = np.int(np.shape(img_in)[1] * shift_x)
    rand_shift_y = np.int(np.shape(img_in)[0] * shift_y)
    # print rand_shift_x, rand_shift_y

    # Shift
    shift_m = np.float32([[1, 0, rand_shift_x], [0, 1, rand_shift_y]])
    img_shift = cv2.warpAffine(img_in, shift_m,
                               (np.shape(img_in)[1], np.shape(img_in)[0]),
                               borderMode=cv2.BORDER_REFLECT)

    return img_shift

def aug_rotation(img_in, params=(-4.0, 4.0, 0.0, 1.0)):
    """
    Rotates the image
    :param img_in: BGR image (cv2 standard)
    :return: BRG image rotated
    """
    cols = np.shape(img_in)[1]
    rows = np.shape(img_in)[0]

    # Random angle (degrees)
    rand_angle = get_trunc_norm(params).rvs()

    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), rand_angle, 1)
    dst = cv2.warpAffine(img_in, m, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    return dst

def aug_perspective(img_in, params=(0.0, 0.01, 0.0, 0.005)):
    """
    Perspective transform of input image
    :param img_in: BGR image (cv2 standard)
    :return: BGR image with perspective transform
    """
    height = np.shape(img_in)[0]
    width = np.shape(img_in)[1]

    distribution = get_trunc_norm(params)

    # Points for transformation
    x = int(round(width * distribution.rvs()))
    y = int(round(height * distribution.rvs()))
    px1 = [x, y]

    x = width - int(round(width * distribution.rvs()))
    y = int(round(height * distribution.rvs()))
    px2 = [x, y]

    x = width - int(round(width * distribution.rvs()))
    y = height - int(round(height * distribution.rvs()))
    px3 = [x, y]

    x = int(round(width * distribution.rvs()))
    y = height - int(round(height * distribution.rvs()))
    px4 = [x, y]

    # print px1, px2, px3, px4

    org = np.float32([px1, px2, px3, px4])

    # Destinations
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    m = cv2.getPerspectiveTransform(org, dst)
    img_out = cv2.warpPerspective(img_in, m, (width, height), borderMode=cv2.BORDER_REFLECT)

    return img_out


def bgr2rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
