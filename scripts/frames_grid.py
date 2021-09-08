from params import *
import matplotlib.pyplot as plt
import random
import cv2

# This script creates a frames grid.

plt.style.use('seaborn-white')
NUM_OF_FRAMES_IN_GRID = 5
random_frame_indexes = [random.randrange(len(os.listdir(TRAINING_SET_DIR))) for _ in range(NUM_OF_FRAMES_IN_GRID)]

# frame1.jpg
# vanilla_dict = {'PSNR': 33.225, 'SSIM': 0.954}
# multilayer_dict = {'PSNR': 31.172, 'SSIM': 0.963}
# conv4_dict = {'PSNR': 29.680, 'SSIM': 0.878}
# conv5_dict = {'PSNR': 27.770, 'SSIM': 0.858}
# vae_dict = {'PSNR': 25.900, 'SSIM': 0.900}

# frame2.jpg
vanilla_dict = {'PSNR': 28.593, 'SSIM': 0.931}
multilayer_dict = {'PSNR': 26.648, 'SSIM': 0.919}
conv4_dict = {'PSNR': 28.773, 'SSIM': 0.896}
conv5_dict = {'PSNR': 26.044, 'SSIM': 0.877}
vae_dict = {'PSNR': 25.126, 'SSIM': 0.876}

# frame3.jpg
# vanilla_dict = {'PSNR': 32.362, 'SSIM': 0.944}
# multilayer_dict = {'PSNR': 30.694, 'SSIM': 0.930}
# conv4_dict = {'PSNR': 30.320, 'SSIM': 0.900}
# conv5_dict = {'PSNR': 27.270, 'SSIM': 0.874}
# vae_dict = {'PSNR': 25.586, 'SSIM': 0.901}


# model_names = ["original", "vanilla", "multilayer"]
# model_results = [{}, vanilla_dict, multilayer_dict]

model_names = ["conv4", "conv5", "vae"]
model_results = [conv4_dict, conv5_dict, vae_dict]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, ax in enumerate(axs.flatten()):
    # image = cv2.imread(TRAINING_SET_DIR + "\\frame" + str(random_frame_indexes[i]) + ".jpg")
    image = cv2.imread(os.path.join("..\demo_showcase_results", model_names[i] + "_compressed_frame2.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.sca(ax)
    plt.imshow(image)
    plt.title('{}'.format(model_names[i]))
    if model_names[i] != "original":
        plt.xlabel("PSNR: " + str(model_results[i]["PSNR"]) + ", SSIM: " + str(model_results[i]["SSIM"]))
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])

plt.tight_layout()
plt.show()
