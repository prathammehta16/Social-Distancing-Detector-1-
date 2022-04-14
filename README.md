# Social-Distancing-Detector using Yolo v3
## **Introduction**
COVID-19 was initially reported during late
December, 2020 in Wuhan, China. And within three
months, it had spreaded in more tha 114 countries and
caused more than 3000 deaths and then WHO declared
this as a pandemic.According to a report by
W.H.O, practicing social distancing and wearing mask
is an important measure to slow down the spread of
virus as individuals with mild indications may
accidentally convey crowd contamination and can
spread the virus to others. Social Distancing implies
that people are suggested that they should maintain
physical distance from one another, reduce close
contact, and thereby reduce the spread of virus. To
automate the task of monitoring social distancing,
concepts of computer vision and deep learning can be
taken into consideration. Computer vision is a field of
artificial intelligence (AI) that enables computers to
derive meaningful information from digital images,
videos and other visual inputs — and take actions or
make recommendations based on that information. The
objects in the image/video are detected in real-time
using YOLO(You only look once), an algorithm
supported by convolutional neural networks which are
employed for the detection and to determine the
distance between the human using clusters of
pedestrians during a neighborhood by grabbing the
feed from a video.

## **Aim**
Our aim is to make an application
that automates the task of monitoring social distancing
among people to hinder the spread of COVID-19.

## **Approach**
To make this project we would use computer
vision(OpenCV) to read the input video file and extract
images from it, then we would use YOLOv3 algorithm
which is based on convolutional neural network which is
a deep learning framework to detect the people in the
video and calculate the distance between the individuals.
Yolo v3 has a 53 layer neural network which is trained on
imagenet. In total it has around 106 layers. Yolo v3 is also
capable of recognising
more than 75 object from the input image or video. The
detailed architecture of Yolo v3 is shown below:
<p align="center">
  <img src="https://github.com/prathammehta16/Social-Distancing-Detector-1-/blob/images/yoloarchitecture.png">
</p>

To predict multiple objects from the image a threshold is
used and to compute class scores logistic regression is used
by Yolo v3. Here we have set the
threshold for predicting people in the video as 50%. So if
a class that has score greater than 0.5, the algorithm will
predict it as people and
bound a box around it. An as the algorithm will predict
multiple bounding boxes per person, we used the concept
of NMS(Non-Max Suppression) boxes which
is the final step in object detection and is used to detect the
most appropriate bounding box for the person.
To calculate distance between people we used the distance
formula from the coordinate geometry i.e. </br>
((x<sub>1</sub>-x<sub>2</sub>)<sup>2</sup>+(y<sub>1</sub>-y<sub>2</sub>)<sup>2</sup>)<sup>0.5</sup>
</br>
Here we set the distance threshold as 50 pixels. So is the
distance between any number of people is found to be less
than 50 pixels then it will be considered as social
distancing violation and a red box will surround that
person. The flow diagram of the project is shown below:
<p align="center">
  <img src="https://github.com/prathammehta16/Social-Distancing-Detector-1-/blob/images/result.png">
</p>

## **Algorithm**
The algorithm used in this project is as follow:
a) Taking video using the VideoCapture object from
OpenCv.</br>
<br>b) Passing the video frame by frame to Yolo v3 network.</br>
<br>c) Detecting the "person" class from the video and then
performing cv2.dnn.blobFromImage() to perform mean
subtraction, scaling and swapping functions.</br>
<br>d) Calculating the co-ordinates of the bounding box.</br>
<br>e) Using Non Maximal Suppression to get the most
appropriate bounding box around the person.</br>
<br>f) Take a variable named 'Violation' and initialize it to
zero.</br>
<br>g) calculate distance between 2 people in the video using
co-ordinate geometry distance formula.</br>
<br>h) If distance < 50 pixels:
<br>Violation++;</br>
