from params import *
import matplotlib.pyplot as plt
import random
import cv2

# This script creates a frames grid.

plt.style.use('seaborn-white')
NUM_OF_FRAMES_IN_GRID = 5
random_frame_indexes = [random.randrange(len(os.listdir(TRAINING_SET_DIR))) for _ in range(NUM_OF_FRAMES_IN_GRID)]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, ax in enumerate(axs.flatten()):
    image = cv2.imread(TRAINING_SET_DIR + "\\frame" + str(random_frame_indexes[i]) + ".jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.sca(ax)
    plt.imshow(image)
    plt.title('frame {}'.format(i+1))
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])

plt.tight_layout()
plt.show()
