import sys
import cv2
import numpy as np
from dataset import CompressDataset
from models import *
from demo_params import *
from utils.PSNR import compare_frames
from utils.blocks import decompose, compose


def print_error_message():
    print("Bad number of arguments.\n" +
          "Run the video demo using the command python video_demo.py <demo_video_index> <model_type> <compress_every>\n" +
          "where demo_video_index is an integer 0-9, model_type is one of vanilla, multilayer, conv4, conv5 and vae.")


if len(sys.argv) == 3:  # we assume arguments are from correct type
    demo_video_index, model_type, compress_every = int(sys.argv[1]), sys.argv[2], 1
    running_psnr, running_ssim = 0.0, 0.0
    demo_video = cv2.VideoCapture(os.path.join(DEMOS_VIDEO_DIR, "demo" + str(demo_video_index) + ".mp4"))
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

        frames, frames_to_compress, frame_counter = [], [], 0
        success, image = demo_video.read()
        while success:
            frames.append(image)
            if frame_counter % compress_every == 0:
                frames_to_compress.append(image)
            success, image = demo_video.read()
            frame_counter += 1

        dataset = CompressDataset(frames_to_compress, False)
        dataloader = torch.utils.data.DataLoader(dataset, **demo_loader_params)

        compressed_frames = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            for j in range(inputs.shape[0]):
                compressed_frame = outputs[j].detach().numpy().reshape((200, 160, 3)) * 255
                compressed_frame = np.clip(compressed_frame, 0, 255)
                compressed_frame = compressed_frame.astype(np.uint8)
                compressed_frames.append(compressed_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        compressed_video = cv2.VideoWriter(os.path.join("demo_results", model_type + "_compressed_video" + str(demo_video_index) + ".mp4"), fourcc, 30, image_dim)
        cv2.startWindowThread()
        cv2.namedWindow("compressed video")
        for i in range(len(frames)):
            if i % compress_every == 0:
                ssim, psnr = compare_frames(compressed_frames[int(i / compress_every)], frames[i])
                running_psnr += psnr
                running_ssim += ssim
                compressed_video.write(compressed_frames[int(i / compress_every)])
                cv2.imshow("compressed video", compressed_frames[int(i / compress_every)])
                cv2.waitKey(30)
            else:
                compressed_video.write(frames[i])
                cv2.imshow("compressed video", frames[i])
                cv2.waitKey(30)

        compressed_video.release()
        cv2.destroyAllWindows()
        print("PSNR: " + str(running_psnr / len(frames_to_compress)) + "\n")
        print("SSIM: " + str(running_ssim / len(frames_to_compress)) + "\n")
        print("Hope you enjoyed, bye bye!")
    elif model_type == "vanilla" or model_type == "multilayer":
        if model_type == "vanilla":
            models = [Vanilla(np.prod(BLOCK_DIMS), VANILLA_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
            for i in range(NUM_OF_BLOCKS):
                models[i].load_state_dict(torch.load(os.path.join(DEMO_VANILLA_MODEL_DIR_PATH, "block" + str(i) + ".pt")))
        else:   # model_type == "multilayer"
            models = [MultiLayer(np.prod(BLOCK_DIMS), MULTILAYER_HIDDEN1_SIZE, MULTILAYER_HIDDEN2_SIZE, MULTILAYER_HIDDEN_LAYER_SIZE, i) for i in range(NUM_OF_BLOCKS)]
            for i in range(NUM_OF_BLOCKS):
                models[i].load_state_dict(torch.load(os.path.join(DEMO_MULTILAYER_MODEL_DIR_PATH, "block" + str(i) + ".pt")))

        frames, frames_to_compress, frame_counter = [], [[] for _ in range(NUM_OF_BLOCKS)], 0
        success, image = demo_video.read()
        while success:
            frames.append(image)
            if frame_counter % compress_every == 0:
                blocks = decompose(image)
                for i in range(NUM_OF_BLOCKS):
                    frames_to_compress[i].append(blocks[i])
            success, image = demo_video.read()
            frame_counter += 1

        datasets = [CompressDataset(frames_to_compress[i], True) for i in range(NUM_OF_BLOCKS)]
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

        compressed_frames = [[] for _ in range(len(frames_to_compress[0]))]
        for block_index in range(NUM_OF_BLOCKS):
            for img_index in range(len(frames_to_compress[0])):
                compressed_frames[img_index].append(blocks_bins[block_index][img_index])

        predicted_frames = []
        for i in range(len(frames_to_compress[0])):
            predicted_frames.append(compose(compressed_frames[i]))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        compressed_video = cv2.VideoWriter(os.path.join("demo_results", model_type + "_compressed_video" + str(demo_video_index) + ".mp4"), fourcc, 30, image_dim)
        cv2.startWindowThread()
        cv2.namedWindow("compressed video")
        for i in range(len(frames)):
            if i % compress_every == 0:
                ssim, psnr = compare_frames(predicted_frames[int(i / compress_every)], frames[i])
                running_psnr += psnr
                running_ssim += ssim
                compressed_video.write(predicted_frames[int(i / compress_every)])
                cv2.imshow("compressed video", predicted_frames[int(i / compress_every)])
                cv2.waitKey(30)
            else:
                compressed_video.write(frames[i])
                cv2.imshow("compressed video", frames[i])
                cv2.waitKey(30)

        compressed_video.release()
        cv2.destroyAllWindows()
        print("PSNR: " + str(running_psnr / len(frames_to_compress[0])) + "\n")
        print("SSIM: " + str(running_ssim / len(frames_to_compress[0])) + "\n")
        print("Hope you enjoyed, bye bye!")
    else:
        print_error_message()

else:
    print_error_message()


