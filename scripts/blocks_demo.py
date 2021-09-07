import cv2
from matplotlib import pyplot as plt
import random
import os
from params import TRAINING_SET_DIR
from utils.blocks import decompose

# This script takes a random fullsize frame and saves a grid of 16 blocks of frames.

NUM_OF_FRAMES = len(os.listdir(TRAINING_SET_DIR))
random_frame_index = random.randrange(NUM_OF_FRAMES)

image_path = TRAINING_SET_DIR + "\\frame" + str(random_frame_index) + ".jpg"
blocks_image_path = "blocks_frame" + str(random_frame_index) + ".jpg"

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
