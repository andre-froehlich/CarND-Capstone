import cv2
import matplotlib.pyplot as plt
import numpy as np

from tl_train_helper import *
from augmentation import *

src = '/Users/jakobkammerer/Learning/carnd/'

# AUGMENTATION
"""
img = cv2.imread(src)


plt.figure()
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(bgr2rgb(augmentation_pipeline(img)))
    plt.axis('off')
plt.show()
"""

# DATASET BALANCE
data = import_data()
balance_dataset(data)
