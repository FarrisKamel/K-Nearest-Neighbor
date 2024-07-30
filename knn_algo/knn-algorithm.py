# This is python program I will implement the knn algorithm for a small data set
# The data set that I used for this is from the following link: https://www.kaggle.com/datasets/gkalpolukcu/knn-algorithm-dataset/data
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to seperate the data with desired label
def dataPreperation(data, label1, label2):
    #  Extract the diagonsis and label column, zip them, convert to array
    data_needed_zip = zip(data["diagnosis"], data[label1], data[label2])
    data_needed = list(data_needed_zip)
    return data_needed

# Function for ploting the data
def plotData(data, label1, label2):
    # Create both arrays we want to plot
    #print(data)
    malignant = [(value1, value2) for label, value1, value2 in data if label == "M"]
    benign = [(value1, value2) for label, value1, value2 in data if label == "B"]
    malignant_x, malignant_y = zip(*malignant)
    benign_x, benign_y = zip(*benign)

    # Add the labels
    plt.figure(figsize=(10, 6))
    plt.scatter(malignant_x, malignant_y, color="red", label="Malignant")  # Red circles for Malignant
    plt.scatter(benign_x, benign_y, color="blue", label="Benign") 

    # Plot
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title('Malignant vs Benign Data')
    plt.legend()
    plt.show()

# K_Nearest Neighbors Algorithm
def kNN(data, unknown_data_point, k=3):

    # Seperate both class
    malignant = [(value1, value2) for label, value1, value2 in data if label == "M"]
    benign = [(value1, value2) for label, value1, value2 in data if label == "B"]

    # Create dict of data
    data_dict = {"M" : malignant, "B": benign}
    #print(data_dict)

    # algo
    distance = []
    for label in data_dict:
        for feature in data_dict[label]:
            # Find the Euclidean distance 
            ed = math.sqrt((feature[0] - unknown_data_point[0])**2 + (feature[1] - unknown_data_point[1])**2)

            # Add the distance with the label
            distance.append((ed, label))

    # sort distance in ascending order
    # Select the first k distances
    distance = sorted(distance)[:k]

    # Find which label is the most frequent and label our unknow data point
    freqM = 0
    freqB = 0
    for value in distance:
        if value[1] == "M":
            freqM += 1
        elif value[1] == "B":
            freqB += 1

    return "M" if freqM > freqB else "B"

# Main function
if __name__ == "__main__":
    #Import data .csv file
    data = pd.read_csv("KNNAlgorithmDataset.csv")
    #print(data.head())
    
    # Call function for data preperation ---> takes in the data and the 2 labels of the column we want to extract 
    data_final = dataPreperation(data, "radius_mean" ,"perimeter_mean")
    
    # Call function for data ploting ---> takes in the data and the 2 labels of the column we want to plot
    plotData(data_final, "radius_mean", "perimeter_mean")

    # KNN-Algorithm
    #Take in the new data point from the user 
    new_radius_mean = input("Enter the Radius Mean value: ")
    new_perimeter_mean = input("Enter the Perimeter Mean value: ")
    k = input("K value for KNN: ")
    unknown_data_point = (int(new_radius_mean), int(new_perimeter_mean))
    result = kNN(data_final, unknown_data_point, int(k))

    # Print results
    print("The result for a Redius Mean of {} and a Perimeter mean of {} is: {}".format(new_radius_mean, new_perimeter_mean, result))
    
    # Add result to data and plot again
    new_point = (result, int(new_radius_mean), int(new_perimeter_mean))
    data_final.append(new_point)
    plotData(data_final, "radius_mean", "perimeter_mean")
