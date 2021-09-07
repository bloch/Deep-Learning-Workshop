import sys
from models import *
from dataset import CompressDataset
import numpy as np
from demo_params import *
from utils.PSNR import compare_frames
from utils.blocks import *


def print_error_message():
    print("Bad number of arguments.\n" +
          "Run the frame demo using the command python frame_demo.py <demo_frame_index> <model_type>\n" +
          "where demo_frame_index is an integer 0-4, model_type is one of vanilla, multilayer, conv4, conv5, vae.")


if len(sys.argv) == 3:  # we assume arguments are from correct type
    demo_frame_index, model_type = int(sys.argv[1]), sys.argv[2]
    image_path = os.path.join("demo_images", "frame" + str(demo_frame_index) + ".jpg")
    compressed_image_path = os.path.join("demo_results", model_type + "_compressed_frame" + str(demo_frame_index) + ".jpg")
    original_image = cv2.imread(image_path)
    if model_type == "conv4" or model_type == "conv5" or model_type == "vae":
        if model_type == "conv4":
            model = ConvAutoencoder4()
            model.load_state_dict(torch.load(DEMO_CONV4_MODEL_PATH))
        if model_type == "conv5":
            model = ConvAutoencoder5()
            model.load_state_dict(torch.load(DEMO_CONV5_MODEL_PATH))
        if model_type == "vae":
            model = VAE()
            model.load_state_dict(torch.load(DEMO_VAE_MODEL_PATH))

        compressed_frames = []
        dataset = CompressDataset([original_image], False)
        loader = torch.utils.data.DataLoader(dataset, **demo_loader_params)
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            for j in range(inputs.shape[0]):
                compressed_frame = outputs[j].detach().numpy().reshape((200, 160, 3)) * 255
                compressed_frame = np.clip(compressed_frame, 0, 255)
                compressed_frame = compressed_frame.astype(np.uint8)
                compressed_frames.append(compressed_frame)

        compressed_frame = compressed_frames[0]
        cv2.imwrite(compressed_image_path, compressed_frame)
        cv2.imshow("compressed image", compressed_frame)
        cv2.waitKey()
        ssim, psnr = compare_frames(compressed_frame, original_image)
        print("The compressed image was saved in " + compressed_image_path + ".")
        print("PSNR: " + str(psnr) + "\n")
        print("SSIM: " + str(ssim) + "\n")
        print("Hope you enjoyed, bye bye!")
    else:   #blocks..
        if model_type == "vanilla" or model_type == "multilayer":
            if model_type == "vanilla":
                models = [Vanilla(np.prod(BLOCK_DIMS), VANILLA_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
                for i in range(NUM_OF_BLOCKS):
                    models[i].load_state_dict(torch.load(os.path.join(DEMO_VANILLA_MODEL_DIR_PATH ,"block" + str(i) + ".pt")))
            else:
                models = [MultiLayer(np.prod(BLOCK_DIMS), MULTILAYER_HIDDEN1_SIZE, MULTILAYER_HIDDEN2_SIZE, MULTILAYER_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
                for i in range(NUM_OF_BLOCKS):
                    models[i].load_state_dict(torch.load(os.path.join(DEMO_MULTILAYER_MODEL_DIR_PATH , "block" + str(i) + ".pt")))

            blocks = decompose(original_image)
            datasets = [CompressDataset([blocks[i]], True) for i in range(NUM_OF_BLOCKS)]
            dataloaders = [torch.utils.data.DataLoader(datasets[i], **demo_loader_params) for i in range(NUM_OF_BLOCKS)]

            compressed_blocks = []
            for block_index in range(NUM_OF_BLOCKS):
                for i, data in enumerate(dataloaders[block_index], 0):
                    inputs, labels = data
                    outputs = models[block_index](inputs)
                    for j in range(inputs.shape[0]):
                        compressed_block = outputs[j].detach().numpy().reshape((50, 40, 3)) * 255
                        compressed_block = np.clip(compressed_block, 0, 255)
                        compressed_block = compressed_block.astype(np.uint8)
                        compressed_blocks.append(compressed_block)

            #compressed_frames should have by now 16 blocks
            compressed_frame = compose(compressed_blocks)
            cv2.imwrite(compressed_image_path, compressed_frame)
            cv2.imshow("compressed image", compressed_frame)
            cv2.waitKey()
            ssim, psnr = compare_frames(compressed_frame, original_image)
            print("The compressed image was saved in " + compressed_image_path + ".")
            print("PSNR: " + str(psnr) + "\n")
            print("SSIM: " + str(ssim) + "\n")
            print("Hope you enjoyed, bye bye!")
        else:
            print_error_message()
else:
    print_error_message()
