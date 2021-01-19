# Using Deep Learning to Detect Lane Lines

I developed an algorithm that can detect lane lines when driving in the CARLA open source simulator.

## Table of contents
- [Dataset](#dataset)
- [Algorithm](#algorithm)
- [Setup](#setup)
- [Credits](#credits)
- [Future plans](#future-plans)

## Dataset

I got the dataset from https://www.kaggle.com/thomasfermi/lane-detection-for-carla-driving-simulator. It was on kaggle so the data was somewhat processed. It consisted of around 3075 images which where each (640,480) RGB images. I initally did try training the model in the 3k images but I didn't seem to get much luck. 

I initially decided to use ImageDataGenerator to pass in the data to the model but that did not seem to work at all. I knew that albemutations was a potential option for image augmentation and to generate more images but I eventually decided to use PIL to convert the image to grayscale, flip it vertically and horizontal, and rotate the images. 

After doing this, I got around 23376 images to work with. I kept the image size at (128,128) since that's what seemed to work best and it did not drain my computer. With the rotation operations I performed, I decided to rotate the image every 5 degrees until it hit the threshold max. I kept the threshold at around 25 degrees resulting in 4 additional images (5, 10, 15, 20). I also added an option to add a threshold value for the pixel values i.e. if pixel_value < threshold, pixel_value = 0. I decided to add this filter to see if the model would do better by getting rid of the noise (background, etc.). I did not find much of a difference in performance whcih I found especially surprising since I had assumed that having more pixels that are black can increase model performance.

I also performed a validation split (of around 0.2). I did not want to play around with this value since it had worked for me and I didn't think it was worth the time to actually play around with this.

These are the shapes of my following data before importing them into the model: X_train: (23376, 128, 128, 1), X_val: (1224, 128, 128, 1), X_test: (1032, 128, 128, 1), y_train: (23376, 128, 128, 1), y_val: (1224, 128, 128, 1), y_test: (1032, 128, 128, 1)

Note: I already have preprocessed the data which you can find in dataset.zip. Download that, and you can directly import it onto your model if you'd like to.

## Algorithm

I used the U-Net model which is a model that was made for biomedical image segmentation by teh University of Freiburg.

![U-net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

The model follow's an encoder-decoder type network where it applies convolutions, maxpooling, and dropout layers. After that, it uses Upsampling to upsample the image and return the output image of the dataset. 

## Setup

Tensorflow == 2.4.0
Keras 

## Credits

@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}

## Future Plans

Result:
![alt text](output_video.gif)


tetet
