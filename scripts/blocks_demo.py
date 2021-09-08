import cv2
from matplotlib import pyplot as plt
import os
from utils.blocks import decompose

# This script takes a random fullsize frame and saves a grid of 16 blocks of frames.

image_path = os.path.join("..\demo_images", "frame6.jpg")
blocks_image_path = "blocks_frame6.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blocks = decompose(image)

_, axarr = plt.subplots(4, 4)

for i in range(4):
    for j in range(4):
        axarr[i, j].imshow(blocks[4*i + j])
        axarr[i, j].set_xticks([])
        axarr[i, j].set_xticklabels("")
        axarr[i, j].set_yticks([25])
        axarr[i, j].set_yticklabels([str(4*i+j)])

plt.savefig(blocks_image_path)
