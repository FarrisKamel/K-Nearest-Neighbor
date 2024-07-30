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
