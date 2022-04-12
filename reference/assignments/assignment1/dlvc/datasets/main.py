import sys
import cv2
import matplotlib.pyplot as plt
from reference.assignments.assignment1.dlvc.datasets.pets import PetsDataset

p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",1)
s=p[3]
image=s[1]
#final_image = plt.imshow(image)
#plt.show()
#RGB - Blue
b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0


# RGB - Blue
#cv2.imshow('B-RGB', b)

# RGB - Green
#cv2.imshow('G-RGB', g)

# RGB - Red
cv2.imshow('R-RGB', r)

cv2.waitKey(0)