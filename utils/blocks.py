import cv2


# decompose 200x160 frame to 16 blocks of 50x40
def decompose(image):
    blocks = []
    for i in range(4):
        for j in range(4):
            block = image[i * 50:i * 50 + 50, j * 40: j * 40 + 40]
            blocks.append(block)
    return blocks


# compose 16 blocks of 50x40 to 200x160 frame
def compose(images_array):
    slices = []
    for i in range(4):
        image1 = cv2.hconcat([images_array[4 * i], images_array[4 * i + 1]])
        image2 = cv2.hconcat([images_array[4 * i + 2], images_array[4 * i + 3]])

        slice_i = cv2.hconcat([image1, image2])
        slices.append(slice_i)

    half1 = cv2.vconcat([slices[0], slices[1]])
    half2 = cv2.vconcat([slices[2], slices[3]])

    image = cv2.vconcat([half1, half2])
    return image
