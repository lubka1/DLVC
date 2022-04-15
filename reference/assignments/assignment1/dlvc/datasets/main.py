import sys
import cv2
import numpy as np
from reference.assignments.assignment1.dlvc import ops
from reference.assignments.assignment1.dlvc.batches import BatchGenerator
from PIL import Image
from reference.assignments.assignment1.dlvc.datasets.pets import PetsDataset

p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",1)
s=p[0]
image=s[1]
b = image.copy()
img = b[:,:,::-1]
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


# g = image.copy()
# # set blue and red channels to 0
# g[:, :, 0] = 0
# g[:, :, 2] = 0
#
# r = image.copy()
# # set blue and green channels to 0
# r[:, :, 0] = 0
# r[:, :, 1] = 0

cv2.imshow('B-RGB', img)
cv2.waitKey(0)

