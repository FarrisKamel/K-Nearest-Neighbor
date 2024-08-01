# K-Nearest-Neighbor

Task: Using KNN and CV, calculate the distance of a car from a Point Of Interest in real time as a video is player.

Motivation: This idea was sparked up because I was wondering what the camera's above the traffic lights do. Some are used to detect if a car passes a certain point and fines them. This is my implementation of that.

## Directory: knn_algo
Before starting the main project, I wanted to freshen up my knowledge on KNN's. In this directory, you will find a data set from the following link: https://www.kaggle.com/datasets/gkalpolukcu/knn-algorithm-dataset/data This dataset provides a 30 different labels for classification. That is different information about a patient's tumor and whether the tumor is Malignant or Bengin. 

I begin by preproessesing the data, that is extracting 2 labels that I wanted to work with their classification: radius_mean and perimeter mean. I then plot the results in a scatter plot for the user to visualize the data. This gives the user an idea of the distribution we are working with. I then ask the user for a new radius_mean value, a new perimeter_mean value, and a k value. These values are then fed into the KNN algorithm and return a classification for the data point. This new data is added to the data and plotted again for the user to see. 

To run:
    
    cd knn_algo
    python3 knn-algorithm.py

## Directory: project
In this directory, you will find my project. The following files can be found in this directory:
* output.mp4
* project.py
* tracker.py
* yolov9c.pt

### Overview
"output.mp4" is a video I took with my phone. It is in Manhattan. I scaled it down so processing is faster. "tracker.py" is a file used to tracker the cars after detection is made. "yolov9c.pt" is a model provided by YOLO for object detection. We focused specifically on the car detection. "project.py" is the project itself. It has comments to document what is happening throughout the code.

### Reflection
This project was interesting. It showed me the different aspects that go into  detection. However there is changes that I would make which will make the software better. For example, the video was caputure some a high angle and tried to capture 2 traffics lights. The video was also taken by hand and therefore not static.  This means that the reference point (the crosswalk lines) was moved as the video progresses. This is not ideal because it makes the measurement inaccurate. The program is also pretty slow. That is because computation is done on the cpu. Threading  would benefit this program but I didn't include it to not complicate things. 

### How to Run
Make sure that all the files are on the same directory. Then run this command:

    python3 project.py
