


# Deep Learning Workshop

This repo contains our workshop's code and demos. Our workshop is about image compression via deep learning. We implemented several different models for image compression and we used that models for video compression as well.

# Dataset

Fullsize(200x160 resolution) examples

![dataset_examples](https://user-images.githubusercontent.com/40773674/132571126-039ee027-fdb3-4295-b40b-66ba4ecafc04.png)


Blocks(50x40 resolution per block) example

![blocks_frame6](https://user-images.githubusercontent.com/40773674/132466274-eb7b7da7-d7c7-4cf4-8a64-641118f074bd.jpg)

Link to dataset videos(original captured 1920x1080): https://drive.google.com/drive/folders/18JMDnRvgXlI3wSm0Un_NZf8baUxrrHx4?usp=sharing

Link to short (original) demo videos(200x160): https://drive.google.com/drive/folders/12I6lkQK_qhdikqtRK2dXcNwzlGLiEZi5?usp=sharing


# Demo Showcase

-------------

## Demo Frames Compression

In the next demos, we used our image compression models on single frames. We show the original frame, and the reconstructed images by each of the models.
The model type is written above the reconstructed frame, and below the PSNR & SSIM ratings.

-------------

### Frame Compression Demo 1 (frame1.jpg)

The results of compression of frame1.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 1 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame1_part1](https://user-images.githubusercontent.com/40773674/132501789-2f118516-e139-43da-9fb8-63ad1e465bbe.png)

![frame1_part2](https://user-images.githubusercontent.com/40773674/132501798-246883c4-a815-4b81-b813-1f9b9c4e537c.png)


-------------

### Frame Compression Demo 2 (frame2.jpg)

The results of compression of frame2.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 2 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame2_part1](https://user-images.githubusercontent.com/40773674/132506277-f20142ea-e1b2-4dad-8433-8633e1af75a1.png)

![frame2_part2](https://user-images.githubusercontent.com/40773674/132506283-e746658d-4987-4de9-821b-5a3220d2e635.png)



-------------

### Frame Compression Demo 3 (frame3.jpg)

The results of compression of frame3.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 3 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame3_part1](https://user-images.githubusercontent.com/40773674/132502683-43c1316a-2f23-4109-b667-05cc9bef5774.png)

![frame3_part2](https://user-images.githubusercontent.com/40773674/132502690-32841a2d-c121-4ce2-8603-3d03d6d962e8.png)



-------------

## Demo Videos Compression

In the next demos, we used our image compression models on videos. We compressed every frame of the video and saved the reconstruction so it is in fact the compressed reconstruction achieved by the models we trained.


-------------

### Video Compression Demo 1 (demo1.mp4)

The results of compression of demo1.mp4 (from the demo_videos directory) by the command
   
      python video_demo.py 1 <model_type>   
      
where <model_type> is written in the video name.

Link to videos(original & compressed version of each model): https://drive.google.com/drive/folders/1zaPk7vQ-wsh9XeZFB8gR5C4gGKoOI_Mj?usp=sharing

#### A table that concludes the PSNR & SSIM results on demo1.mp4


| Model name | PSNR | SSIM |
| ------------ | ------------ | ------------ |
| vanilla | 29.032 | 0.900 |
| multilayer | 27.749 | 0.896 |
| conv4 | 28.357 | 0.869 |
| conv5 | 26.946 | 0.862 |
| vae | 25.072 | 0.863 |


-------------

### Video Compression Demo 2 (demo2.mp4)

The results of compression of demo2.mp4 (from the demo_videos directory) by the command
   
      python video_demo.py 2 <model_type>   
      
where <model_type> is written in the video name.


Link to videos(original & compressed version of each model): https://drive.google.com/drive/folders/1n_g8e0mICJ8Lbv_mvhGbn3DItXf_t-yW?usp=sharing

#### A table that concludes the PSNR & SSIM results on demo2.mp4


| Model name | PSNR | SSIM |
| ------------ | ------------ | ------------ |
| vanilla | 31.253 | 0.921 |
| multilayer | 31.041 | 0.921 |
| conv4 | 29.993 | 0.895 |
| conv5 | 28.934 | 0.873 |
| vae | 27.923 | 0.892 |



-------------

### Video Compression Demo 3 (demo3.mp4)

The results of compression of demo3.mp4 (from the demo_videos directory) by the command
   
      python video_demo.py 3 <model_type>   
      
where <model_type> is written in the video name.


Link to videos(original & compressed version of each model): https://drive.google.com/drive/folders/1_q6OBaXLP-ZaudCHFWAlt5mr48ctPXLd?usp=sharing

#### A table that concludes the PSNR & SSIM results on demo3.mp4

Model name | PSNR | SSIM 
------------ | -------------  | -------------
| vanilla | 29.967 | 0.896 |
| multilayer | 28.611 | 0.900 |
| conv4 | 28.172 | 0.837 |
| conv5 | 27.556 | 0.849 |
| vae | 25.881 | 0.871 |


-------------


# Live Demo Instructions

### Remark: The downloads & installations required in order to run a live demo will take several minutes and a descent amount of storage(~4.5GB), and therefore we provided the above demo examples for convience(so one doesn't have to run the live demo to see some results).


1. Download all files in the repository.
   
   This can be done by clicking on the green button of 'Code' and using the 'Download ZIP' option.
   
   Then, extract the contents of the ZIP. This will create a new directory named 'Deep-Learning-Workshop-master' which includes all the code and some of the models(vanilla, conv4    and conv5 models). The models are located in Deep-Learning-Workshop-master/demo_models directory.
   
   Remark: Multilayer and VAE models are heavy and were not uploaded to this git repo and are stored on Google drive. If one wants to use the multilayer and VAE models in the live    demo, he should download them from the drive and locate them in the following way: 
      - the vae.pt file should be at the demo_models directory(the relative path should be 'Deep-Learning-Workshop-master/demo_models/vae.pt').
        Download (and extract from the zip file) the vae model(vae.pt file) from the following link: https://drive.google.com/file/d/1qfrRyRIYVIBpiMW7WV3di8Q2KdroFK7F/view?usp=sharing.
      - the multilayer model(consists of 16 .pt files) should be a sub-directory at the demo_models directory, and the relative path should
        be 'Deep-Learning-Workshop-master/demo_models/multilayer/'.
        Download (and extract from the zip files) the multilayer models dir from https://drive.google.com/drive/folders/1SRe9r7zjkBkO0VPOmzQNrdnJaZktdSiE?usp=sharing.
        This might be downloaded in a seperate zip files, and you have to merge all the .pt files to the 'demo_models/multilayer/ directory.
        
   
2. The demo_models directory should look like(after downloading the extra models from the drive):
   
      ![image](https://user-images.githubusercontent.com/40773674/132831062-27218c3e-9910-4bb7-949a-467c0bceee3f.png)
      
   If multilayer and VAE models were not downloaded, it should look like the demo_models directory in this repo.

   Remark: This is an additional link to a google drive directory with all models(this is actually demo_models dir):

   https://drive.google.com/drive/folders/1-unbekPRk1qUICMMxzVB59wNmXpQp4RG?usp=sharing
   
3. Optional(if installation problems occur): Open a terminal within the Deep-Learning-Workshop-master directory and run the following command to install required packages:

            pip install -r requirements.txt
            
4. Now everything is ready for running the demo.
   
   - For using the a single frame demo, one should open a terminal and run the following command:
   
            python frame_demo.py <frame_index> <model_type>
    
      where frame_index is an integer between 0-9 representing which frame to compress from the demo_images directory, and model_type is one of the following options: vanilla,           multilayer, conv4, conv5, vae, and this however represents what model the user wishes to use for compression.
   
      For example:
   
            python frame_demo.py 1 conv4
   
      The frame demo will show the original image on the left and the reconstruction from the compressed represenation on the right.
   
      It also saves the reconstructed image in the demo_results directory with the name '<demo_type>_compressed_frame<frame_index>.jpg'.
   
   - For using the video demo, one should open a terminal and run the following command:
   
            python video_demo.py <video_index> <model_type>
      
      where video_index is an integer between 0-9 representing which video to compress from the demo_videos directory, and model_type is one of the following options: vanilla,           multilayer, conv4, conv5, vae, and this however represents what model the user wishes to use for compression.
   
      For example:
   
            python video_demo.py 3 vanilla
      
      The video demo will show the the reconstructed video.
   
      It also saves the reconstructed video in the demo_results directory with the name '<demo_type>_compressed_video<video_index>.jpg'.
      
 Both demos will print PSNR and SSIM (compression performance measurements) of the compressed frame/video.
