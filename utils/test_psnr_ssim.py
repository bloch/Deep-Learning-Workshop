import sys
import cv2
import numpy as np
from dataset import Dataset
from models import *
from params import *
from utils.PSNR import compare_frames
from utils.blocks import compose



if len(sys.argv) == 2:  # we assume arguments are from correct type
    model_type = sys.argv[1]
    running_psnr, running_ssim = 0.0, 0.0
    frames = []
    for i in range(len(os.listdir(TEST_SET_DIR))):
        img = cv2.imread(os.path.join(TEST_SET_DIR, "frame" + str(i) + ".jpg"))
        frames.append(img)
    if model_type == "conv4" or model_type == "conv5" or model_type == "vae":  # add or ... on all Models that don't use blocks
        dataset = Dataset(0, TEST_SET_DIR, False)
        test_loader = torch.utils.data.DataLoader(dataset, **demo_loader_params)
        criterion = nn.MSELoss()
        if model_type == "conv4":
            model = ConvAutoencoder4()
        if model_type == "conv5":
            model = ConvAutoencoder5()
        if model_type == "vae":
            model = VAE()

        model.load_state_dict(torch.load(model.path))
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            outputs = model(inputs)
            output = outputs[0].detach().numpy().reshape((200, 160, 3)) * 255
            output = np.clip(output, 0, 255)
            output = output.astype(np.uint8)
            ssim, psnr = compare_frames(output, frames[i])
            running_psnr += psnr
            running_ssim += ssim
        print("Average PSNR of the compressed frames: " + str(running_psnr / dataset.length) + "\n")
        print("Average SSIM of the compressed frames: " + str(running_ssim / dataset.length) + "\n")

    elif model_type == "vanilla" or model_type == "multilayer":

        if model_type == "vanilla":
            models = [Vanilla(np.prod(BLOCK_DIMS), VANILLA_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
            for i in range(NUM_OF_BLOCKS):
                models[i].load_state_dict(
                    torch.load(os.path.join(DEMO_VANILLA_MODEL_DIR_PATH, "block" + str(i) + ".pt")))
        else:  # model_type == "multilayer"
            models = [MultiLayer(np.prod(BLOCK_DIMS), MULTILAYER_HIDDEN1_SIZE, MULTILAYER_HIDDEN2_SIZE,
                                 MULTILAYER_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
            for i in range(NUM_OF_BLOCKS):
                models[i].load_state_dict(
                    torch.load(os.path.join(DEMO_MULTILAYER_MODEL_DIR_PATH, "block" + str(i) + ".pt")))


        datasets = [Dataset(0, TEST_SET_BLOCKS_DIR + "\\block" + str(i), True) for i in range(NUM_OF_BLOCKS)]
        dataloaders = [torch.utils.data.DataLoader(datasets[i], **demo_loader_params) for i in range(NUM_OF_BLOCKS)]

        blocks_bins = [[] for _ in range(NUM_OF_BLOCKS)]
        for block_index in range(NUM_OF_BLOCKS):
            for i, data in enumerate(dataloaders[block_index], 0):
                inputs, labels = data
                outputs = models[block_index](inputs)
                for j in range(inputs.shape[0]):
                    compressed_frame = outputs[j].detach().numpy().reshape((50, 40, 3)) * 255
                    compressed_frame = np.clip(compressed_frame, 0, 255)
                    compressed_frame = compressed_frame.astype(np.uint8)
                    blocks_bins[block_index].append(compressed_frame)

        compressed_frames = [[] for _ in range(datasets[0].length)]
        for block_index in range(NUM_OF_BLOCKS):
            for img_index in range(datasets[0].length):
                compressed_frames[img_index].append(blocks_bins[block_index][img_index])

        predicted_frames = []
        for i in range(datasets[0].length):
            predicted_frames.append(compose(compressed_frames[i]))

        for i in range(len(frames)):
            ssim, psnr = compare_frames(frames[i], predicted_frames[i])
            running_psnr += psnr
            running_ssim += ssim
        print("Average PSNR of the compressed frames: " + str(running_psnr / len(frames)) + "\n")
        print("Average SSIM of the compressed frames: " + str(running_ssim / len(frames)) + "\n")
