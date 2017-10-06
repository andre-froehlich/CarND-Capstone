import cv2
import matplotlib.pyplot as plt
import numpy as np

from tl_train_helper import *
from augmentation import *

src = '/Users/jakobkammerer/Google Drive/Happy Robots/train/'

test_data = import_data(source='*/')
#test_data = get_dataset(test_data)
#print(test_data)
test_data = balance_dataset(test_data)

#print(test_data)

img = cv2.imread('/Users/jakobkammerer/Google Drive/Happy Robots/train/simulator/0-red/data000006.png')
img = resize_img(img)
print(img.shape)
plt.imshow(img)
plt.show()