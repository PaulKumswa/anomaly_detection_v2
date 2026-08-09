# Steel Surface Defect Classification

This project explored using a neural network to detect defects in steel manufacturing

## Project Overview

The project used a neural net and image processing pipelines to first train and test the model, and then display its results using streamlit app.

It classifies images into 4 different types of defects, and a category with no defects.

## Dataset Preparation and Splits

I used a custom script called data_prep.py to place the images into their correct folders for classification. This script used the original dataset and the train.csv file to do this.

I only used images with only 1 defect label, or none at all. This was for simplicity. I could not find a way to handle the multiple defect class of images without changing the core of the model.

The model uses stratification, a fixed random seed of 42, and 70%/15%/15% train/validation/test ratios.

## Data Preprocessing

The preprocessing had traning and validation pipelines. All images were resized to 256x256. In order to augement training, images were given a random flip probability and also a random brightness contrast. The flip and brightness were not added to validation, for the sake of reproducibility.

## Model Architecture

Model info: It has abou 103k params, and convolutional blocks with 32, 64, and 128 output channels.

## Training

The training loop used a CrossEntropyLoss loss function, an Adam optimizer, learning rate of 0.001, and a batch size of 32. I used onl a CPU for training and the model ran for 20 epochs. 

Model checkpoints only happen when the model improves. 

Total tranin gtime was about 4 hours for me, and epoch 17 had the best validation accuracy. My final validation accuracy was 0.791.

## Results and Inference

I was able to run the application in streamlitapp, with about 60ms of inference time. my final model accuracy was 79.1%.

## Challenges and Learnings

I enjoyed working with pytorch and seeing the inner workings of a NN. Deciding what to do with multi-label images was one of the more-difficult parts of this. Also, while CPU training took a long time, I was happy with the results.
