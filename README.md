

https://user-images.githubusercontent.com/40773674/132379816-0bd0a04e-6921-4983-a7fd-32437a56d930.mp4








# Deep Learning Workshop

This repo contains our workshop's code and demos. Our workshop is about image compression via deep learning. We implemented several different models for image compression and we used that models for video compression as well.

# Dataset examples

![alt text](https://github.com/bloch/Deep-Learning-Workshop/blob/master/Visualizations/dataset_examples.png?raw=true)

https://user-images.githubusercontent.com/40773674/132379119-25abc140-d95b-45dd-ae89-0f023fe14f79.mp4
# Demos


# Live Demo instructions


1. Download all files in the repository(specifically: dataset.py, models.py, demo_params.py, and all py files in utils folder).
   
   This can be done by clicking on the green button of 'Code' and using the 'Download ZIP' option.
   
   Then, extract the contents of the ZIP. This will create a new directory named 'Deep-Learning-Workshop-master' which includes all the code.
   
2. Download the demo models from the following link:

   https://drive.google.com/drive/folders/1-unbekPRk1qUICMMxzVB59wNmXpQp4RG?usp=sharing
   
   and extract the contents(one directory named demo_models with all the models included in it) to the folder Deep-Learning-Workshop-master.
   
   This should look like Deep-Learning-Workshop-master\demo_models.
   
3. Optional(if installations problems occur): Open a terminal within the Deep-Learning-Workshop-master directory and run the following command to install required packages:

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
