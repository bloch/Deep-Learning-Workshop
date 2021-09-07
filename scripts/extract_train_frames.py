import cv2
from params import *
from utils.blocks import decompose

# saves full-size(160x200) training frames to TRAINING_SET_DIR
# saves block-divided(40x50) training frames blocks to TRAINING_SET_BLOCKS_DIR\\block<block_index>

frame_counter = 0
for video_counter in range(1, NUM_OF_TRAIN_VIDEOS + 1):
    vidcap = cv2.VideoCapture(videos_dir_path + "\\video" + str(video_counter) + ".mp4")
    success, image = vidcap.read()
    while success:
        image = image[5:1035, 545:1375]
        image = cv2.resize(image, image_dim, interpolation=cv2.INTER_LANCZOS4)

        blocks = decompose(image)
        for i, block in enumerate(blocks):
            cv2.imwrite(TRAINING_SET_BLOCKS_DIR + "\\block" + str(i) + "\\frame" + str(frame_counter) + ".jpg", block)

        cv2.imwrite(TRAINING_SET_DIR + "\\frame" + str(frame_counter) + ".jpg", image)

        for _ in range(skip_every):
            success, image = vidcap.read()

        frame_counter += 1
