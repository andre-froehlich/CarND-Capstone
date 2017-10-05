import cv2
import matplotlib.pyplot as plt
import numpy as np

from augmentation import *
from tl_train_helper.py import *

src = '/Users/jakobkammerer/Google Drive/Happy Robots/TrafficLightData_real/data000155.png'

img = cv2.imread(src)


plt.figure()
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(bgr2rgb(augmentation_pipeline(img)))
    plt.axis('off')
plt.show()
