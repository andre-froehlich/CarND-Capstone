import cv2
import numpy as np


def augmentation_pipeline(img):
    if np.random.randint(2) == 1:
        return img

    # Brightness 50/50
    if np.random.randint(2) == 1:
        img = aug_brightness(img)
    # Color 50/50
    if np.random.randint(2) == 1:
        img = aug_color(img)
    # Shadow 50/50
    if np.random.randint(2) == 1:
        img = aug_shadow(img)
    # Blur 50/50
    if np.random.randint(2) == 1:
        img = aug_blur(img)
    # Shift or Rotation
    if np.random.randint(2) == 1:
        if np.random.randint(2) == 1:
            img = aug_shift(img)
        else:
            img = aug_rotation(img)
    # Perspective Transform 50/50
    if np.random.randint(2) == 1:
        img = aug_perspective(img)
    # Flip 50/50
    if np.random.randint(2) == 1:
        img = aug_flip(img)

    return img

def aug_brightness(img_in):
    """
    Augment picture in brightness
    :param img_in: image in BGR (openCV standard)
    :return: image in BGR, augmented in brightness
    """
    # Change to HSV color space and transform to float64 array
    img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    img_out = np.array(img_out, dtype=np.float64)
    # Set a random number for brightness adjustment and adjust
    rand_bright = 0.5 + np.random.uniform()
    img_out[:, :, 2] = img_out[:, :, 2] * rand_bright
    # set every value higher than 255 to 255
    img_out[:, :, 2][img_out[:, :, 2] > 255] = 255
    img_out[:, :, 2][img_out[:, :, 2] < 0] = 0
    # Transform back (unit8 and BRG)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
    return img_out


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


def aug_blur(img_in):
    """
    Blurs the image
    :param img_in: BGR image (cv2 standard)
    :return: BGR image with blur
    """
    rand_kernel_size = np.random.randint(low=5, high=21)
    rand_kernel = (rand_kernel_size, rand_kernel_size)

    return cv2.blur(img_in, rand_kernel)


def aug_shift(img_in):
    """
    Shifts picture in x and y by random (max. 15 % of pixels)
    :param img_in: BGR image (cv2 standard)
    :return: BGR image shifted
    """
    # Percentage
    # rand_shift_x = np.random.randint(low=1, high=31) - 15
    # rand_shift_y = np.random.randint(low=1, high=31) - 15

    shift_x = np.random.normal(0.0, 0.05)
    shift_y = np.random.normal(0.0, 0.05)

    # Pixels
    rand_shift_x = np.int(np.shape(img_in)[1] * shift_x)
    rand_shift_y = np.int(np.shape(img_in)[0] * shift_y)
    # print rand_shift_x, rand_shift_y

    # Shift
    shift_m = np.float32([[1, 0, rand_shift_x], [0, 1, rand_shift_y]])
    img_shift = cv2.warpAffine(img_in, shift_m,
                               (np.shape(img_in)[1], np.shape(img_in)[0]))

    return img_shift

def aug_flip(img_in):
    """
    flips image vertically
    :param img_in: BGR image (cv2 standard)
    :return: BRG image flipped
    """
    return np.fliplr(img_in)


def aug_color(img_in):
    """
    Manipulates color channelwise by random (channel and value)
    :param img_in: BGR image (cv2 standard)
    :return: BGR image manipulated in color
    """
    # Convert to float64 array
    img_out = np.array(img_in, dtype=np.float64)

    # Random Color Channel and random manipulation
    rand_color = np.random.randint(3)
    rand_manipulation = np.random.randint(low=-20, high=21)

    # Manipluation
    img_out[:, :, rand_color] = img_out[:, :, rand_color] + rand_manipulation
    img_out[:, :, rand_color][img_out[:, :, rand_color] > 255] = 255
    img_out[:, :, rand_color][img_out[:, :, rand_color] < 0] = 0

    # Convert back and return
    return np.array(img_out, dtype=np.uint8)


def aug_rotation(img_in):
    """
    Rotates the image
    :param img_in: BGR image (cv2 standard)
    :return: BRG image rotated
    """
    cols = np.shape(img_in)[1]
    rows = np.shape(img_in)[0]

    # Random angle (degrees)
    rand_angle = np.random.randint(low=-10, high=11)

    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), rand_angle, 1)
    dst = cv2.warpAffine(img_in, m, (cols, rows))

    return dst


def aug_perspective(img_in):
    """
    Perspective transform of input image
    :param img_in: BGR image (cv2 standard)
    :return: BGR image with perspective transform
    """
    height = np.shape(img_in)[0]
    width = np.shape(img_in)[1]

    # Set maximum grab for perspective transform
    percentage_width = int(0.1 * width)
    percentage_height = int(0.1 * height)

    # Points for transformation
    px1 = [0 + np.random.randint(percentage_width),
           0 + np.random.randint(percentage_height)]
    px2 = [width - np.random.randint(percentage_width),
           0 + np.random.randint(percentage_height)]
    px3 = [width - np.random.randint(percentage_width),
           height - np.random.randint(percentage_height)]
    px4 = [0 + np.random.randint(percentage_width),
           height - np.random.randint(percentage_height)]

    org = np.float32([px1, px2, px3, px4])

    # Destinations
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    m = cv2.getPerspectiveTransform(org, dst)
    img_out = cv2.warpPerspective(img_in, m, (width, height))

    return img_out


def bgr2rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
