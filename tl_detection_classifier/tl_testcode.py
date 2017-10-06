import cv2
import matplotlib.pyplot as plt
import numpy as np

from tl_train_helper import *
from augmentation import *

src = '/Users/jakobkammerer/Learning/carnd/'

test_data = import_data(source='*/')
#test_data = get_dataset(test_data)
#print(test_data)
test_data = balance_dataset(test_data)

print(test_data)


