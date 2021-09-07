import cv2
from params import *

# This scripts creates a 160x200 resolution video from the demo video(video30).

video = cv2.VideoCapture(DEMO_VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_160x200 = cv2.VideoWriter(VIDEO_160x200_PATH, fourcc, 30, image_dim)

success, image = video.read()
while success:
    image = image[5:1035, 545:1375]
    image = cv2.resize(image, image_dim, interpolation=cv2.INTER_LANCZOS4)
    video_160x200.write(image)
    success, image = video.read()

video_160x200.release()
