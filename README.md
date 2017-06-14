### Vehicle Detection Project ###

## **Here I take a deep learning approach which is different with the HOG + SVM in the lecture** ##

[//]: # (Image References)
[image1]: output_images/cnn_output.png
[image2]: output_images/raw_feature_map.png
[image3]: output_images/raw_bb_1.png
[image4]: output_images/raw_hm_1.png
[image5]: output_images/filtered_hm_1.png
[image6]: output_images/labels_map.png
[image7]: output_images/labeled_hm.png
[image8]: output_images/draw_label.png
[image9]: output_images/full_proccess_1.png
[image10]: output_images/full_process_2.png
[image11]: output_images/full_process_3.png


### Deep Learning Model

My best model is available from [model link](https://benk-carnd-public.s3-us-west-2.amazonaws.com/car_model.h5)

#### 1) Full-convolutional neural network
I choose LeNet as my base model since the MNIST photo size (32x32) is very close to our training set size(64x64). Then I change the last couple of full-connected layers in LeNet to convolutional layers.

Traditionally convolutional neural networks end with one or more fully connected layers which will remove the spatial information.

To use CNN for segmentation problem we can transform the full-connected layers in CNN to multi 1x1 convolutional layer. By this design for a 64x64 photo the output of the model should be just 1 float number as the possibility of a car between (0~1). For any other size photo in theory it is similar to grid search 64x64 pixels on the full photo and the output size is size of the input photo divide the last feature map size(in my model it is the output after the second maxpooling layers) in the model.

 My model architect is (train.py line 81):

| LAYER | Description|
| ----- | ---------- |
| lambda layer| Normalize pictures|
| conv layer (64, 5, 5, same, elu)| First conv layer|
| maxpool(4,4,same)| Pooling|
| normalization| Batch normalization|
| dropout| 0.5 |
| conv layer (128, 3, 3, same, elu)|Second conv layer|
| maxpool(2,2,same)|Pooling|
|normalization| Batch normalization|
| dropout| 0.5|
| conv layer (1024, 8, 8, elu)| Replace first FC layer|
|normalization| Batch normalization|
| dropout| 0.25|
| conv layer (100, 1, 1, elu)| Replace second FC layer|
| conv layer (1, 1, 1, sigmoid)|Output node|

Because the problem we have to classify vehicle or non-vehicle so I use `sigmoid` as final output function

#### 2) Training
Before call the fit method on the model I have to flat the multi-layer 1x1 convolutional output to normal Nx1 target. And it should not be part of the final model which means when using the final model for prediction we will not call the flatten method on the model
I use `binary_crossentropy` as the loss function and `adam` as the optimizer again because the problem is a binary classification. The final hyper-params are set in `train.py` line 132-133

#### 3) Training Pipeline
To run the training code,
* First put positive training set photos under `training/positive`
* Put negative training set photos under `training/negative`
* Run `python data_augment.py` to generate training set list csv files
* Run `python train.py` to train the model

#### 4) Notes
I had to save frames from the video and crop cars and negative samples from the frames to add to the training set to get a better result.

#### 5) Model Result
This is some sample result from my CNN classifier

![model-result][image1]

* my model output for first photo: `0.0000023`
* my model output for second photo `0.9999995`

#### 6) h5 model

My model is saved as h5 file and [model link](https://benk-carnd-public.s3-us-west-2.amazonaws.com/car_model.h5)


### Classification and Segmentation

#### 1) Result from DNN classifier
This is some sample output from my neural network model when pass in a region of interest of the frame from the video,

![model-result][image2]

then I scale the output back to original size and draw some bounding with the raw output.

![model-result][image3]

The code is at `segmentation.py` line 25 and 30. I have to find a threshold to filter out some false positive from the classification result, here I use 0.95

#### 2) Create heatmap
To combine the raw bounding boxes, I create a heatmap from the model output above the code is `segmentation.py` line 54

![model-result][image4]

#### 3) Find heatmap thershold
To reduce the false positive I have to filter the heatmap by some threshold at `segmentation.py` line 64

![model-result][image5]

#### 4) Label and Draw bounding boxes
Then I use `scipy.ndimage.measurements` to label my filtered heatmap and draw bounding boxes the code is at line 66 and line 69

![model-result][image7]
![model-result][image8]

### Video Process Pipeline
Put everything together is the video process pipeline which code is in proccessvideo.py. You need to download my model first from [model link](https://benk-carnd-public.s3-us-west-2.amazonaws.com/car_model.h5)
 then run the process pipeline `python proccessvideo.py input_video output_video`.

 Some sample frames transformation:

![model-result][image9]

![model-result][image10]

![model-result][image11]

### Output video
There are still some false positives in my final video also my bounding boxes are not smooth.

If I have more time I will fine-tune the parameters a bit more and add more negative training sets to get better results.

Here's my result

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_jriDJ8PmjU/0.jpg)](https://www.youtube.com/watch?v=_jriDJ8PmjU)

[youtube link](https://youtu.be/_jriDJ8PmjU)

###Discussion

* I struggled a bit to train the model and one of the issues was that training set wasn't shuffled enough. My validation cost didn't decrease stably.
* I still have some false negative in the final output, if more time permitted I will add more negative training set and fine-tune the thresholds to eliminate more false negative.
* My bounding box drawing is not smooth in the video, if more time permitted I will try to add more caching, averaging and caching to make it smoother.
* Lastly thanks for the classmates in slack channels to encourage me to take a deep learning approach. As I was told and read that HOGS could be considered as a special type of convolutional neural network.
