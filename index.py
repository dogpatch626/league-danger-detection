import numpy as np
import argparse
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt


def isolate_red ():

    my_image = cv2.imread("sc.png")
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)

# red color boundary
    lower_red = np.array([100, 20, 20], dtype = np.uint8)
    upper_red = np.array([255, 100 , 100], dtype = np.uint8)

    mask = cv2.inRange(my_image, lower_red, upper_red)
    img = cv2.bitwise_and(my_image, my_image, mask = mask)

    plt.subplot(1,2,1)

    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(my_image)
    plt.show()


isolate_red()
