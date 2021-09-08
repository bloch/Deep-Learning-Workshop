




# Deep Learning Workshop

This repo contains our workshop's code and demos. Our workshop is about image compression via deep learning. We implemented several different models for image compression and we used that models for video compression as well.

# Dataset

Fullsize(200x160 resolution) examples

![alt text](https://github.com/bloch/Deep-Learning-Workshop/blob/master/Visualizations/dataset_examples.png?raw=true)


Blocks(50x40 resolution per block) example

![blocks_frame6](https://user-images.githubusercontent.com/40773674/132466274-eb7b7da7-d7c7-4cf4-8a64-641118f074bd.jpg)

Link to dataset videos: https://drive.google.com/drive/folders/18JMDnRvgXlI3wSm0Un_NZf8baUxrrHx4?usp=sharing

Link to short (original) demo videos: https://drive.google.com/drive/folders/12I6lkQK_qhdikqtRK2dXcNwzlGLiEZi5?usp=sharing


# Demo Showcase

-------------

# Demo Frames Compression

## Frame Compression Demo 1 (frame1.jpg)

The results of compression of frame1.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 1 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame1_part1](https://user-images.githubusercontent.com/40773674/132501789-2f118516-e139-43da-9fb8-63ad1e465bbe.png)

![frame1_part2](https://user-images.githubusercontent.com/40773674/132501798-246883c4-a815-4b81-b813-1f9b9c4e537c.png)


-------------

## Frame Compression Demo 2 (frame2.jpg)

The results of compression of frame2.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 2 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame2_part1](https://user-images.githubusercontent.com/40773674/132506277-f20142ea-e1b2-4dad-8433-8633e1af75a1.png)

![frame2_part2](https://user-images.githubusercontent.com/40773674/132506283-e746658d-4987-4de9-821b-5a3220d2e635.png)



-------------

## Frame Compression Demo 3 (frame3.jpg)

The results of compression of frame3.jpg (from the demo_images directory) by the command
   
      python frame_demo.py 3 <model_type>   
      
where <model_type> is written above the corresponding image. 

![frame3_part1](https://user-images.githubusercontent.com/40773674/132502683-43c1316a-2f23-4109-b667-05cc9bef5774.png)

![frame3_part2](https://user-images.githubusercontent.com/40773674/132502690-32841a2d-c121-4ce2-8603-3d03d6d962e8.png)



-------------

# Demo Videos Compression

In the next demos, we used our image compression models on videos. We compressed every frame of the video and saved the reconstruction so it is in fact the compressed reconstruction achieved by the models we trained.

## Video Compression Demo 1 (demo1.mp4)

Link to videos(original & compressed version of each model): https://drive.google.com/drive/folders/1zaPk7vQ-wsh9XeZFB8gR5C4gGKoOI_Mj?usp=sharing

A table that concludes the PSNR & SSIM results on demo1.mp4


## Video Compression Demo 2 (demo2.mp4)

Link to videos(original & compressed version of each model):

A table that concludes the PSNR & SSIM results on demo3.mp4


## Video Compression Demo 3 (demo3.mp4)

Link to videos(original & compressed version of each model): https://drive.google.com/drive/folders/1_q6OBaXLP-ZaudCHFWAlt5mr48ctPXLd?usp=sharing

A table that concludes the PSNR & SSIM results on demo3.mp4



# Live Demo Instructions


1. Download all files in the repository(specifically: dataset.py, models.py, demo_params.py, and all py files in utils folder).
   
   This can be done by clicking on the green button of 'Code' and using the 'Download ZIP' option.
   
   Then, extract the contents of the ZIP. This will create a new directory named 'Deep-Learning-Workshop-master' which includes all the code.
   
2. Download the demo models(the base directory of the link) from the following link:

   https://drive.google.com/drive/folders/1-unbekPRk1qUICMMxzVB59wNmXpQp4RG?usp=sharing
   
   and extract the contents(one directory named demo_models with all the models included in it) to the folder Deep-Learning-Workshop-master.
   
   This should look like Deep-Learning-Workshop-master\demo_models.
   
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
