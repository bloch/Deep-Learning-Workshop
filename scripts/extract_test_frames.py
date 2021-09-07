import cv2

from params import *
from utils.blocks import decompose

frame_counter = 0
for test_video_index in test_video_indexs:
    test_video = cv2.VideoCapture(videos_dir_path + "/video" + str(test_video_index) + ".mp4")
    success, image = test_video.read()
    while success:
        image = image[5:1035, 545:1375]
        image = cv2.resize(image, image_dim, interpolation=cv2.INTER_LANCZOS4)

        blocks = decompose(image)
        for i, block in enumerate(blocks):
            cv2.imwrite(TEST_SET_BLOCKS_DIR + "/block" + str(i) + "/frame" + str(frame_counter) + ".jpg", block)

        cv2.imwrite(TEST_SET_DIR + "/frame" + str(frame_counter) + ".jpg", image)

        for _ in range(skip_every_validation):
            success, image = test_video.read()
        frame_counter += 1
