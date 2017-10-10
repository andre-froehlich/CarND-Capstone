import cv2
from augmentation import augmentation_pipeline

image = cv2.imread("/media/student/OS/Users/andre/Google Drive/Happy Robots/train 2/simulator/1-yellow/img_time  1507294894503067970_state1.png")

image = augmentation_pipeline(image,
                              brightness=True,
                              color=True,
                              shadow=True,
                              blur=True,
                              shift=True,
                              rot=True,
                              perspective=True,
                              aug_prob=1.0
                              )

cv2.imwrite("generator_output/img.png", image)