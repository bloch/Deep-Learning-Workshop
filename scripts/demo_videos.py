import cv2

from demo_params import DEMOS_VIDEO_DIR
from params import *

# This script creates short demos videos for submission and demo, from video30.

video = cv2.VideoCapture(DEMO_VIDEO_PATH)

for _ in range(START_FRAME):
    _, image = video.read()

for demo_index in range(NUM_OF_DEMO_VIDEOS):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    demo = cv2.VideoWriter(DEMOS_VIDEO_DIR + "\\demo" + str(demo_index) + ".mp4", fourcc, 30, image_dim)
    for frame_index in range(START_FRAME + demo_index*FRAMES_IN_EACH_DEMO, START_FRAME + (demo_index+1)*FRAMES_IN_EACH_DEMO + 1):
        _, image = video.read()
        image = image[5:1035, 545:1375]
        image = cv2.resize(image, image_dim, interpolation=cv2.INTER_LANCZOS4)
        demo.write(image)

    demo.release()
