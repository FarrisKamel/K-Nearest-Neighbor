import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import*
import time
import math

# Function for frame processing 
def processFrame(frame):
    # Get the results from out model
    results = car_model.predict(frame)

    # Get access to the data from and box information
    df = results[0].boxes.data
    df = df.detach().cpu().numpy()
    px = pd.DataFrame(df).astype("float")
    #print(px) # Print to see how it looks like

    # Store all the values in a list
    car_info = []

    # Get all the data points and check that classification is a car
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if "car" in c:
            car_info.append([x1,y1,x2,y2])

    # Use the above information for the tracker and inilize center cooridinates
    car_id = tracker.update(car_info)
    center_x, center_y = 0, 0

    # Add the lines on the road
    cv2.line(frame, (987, 549), (669, 378), (0, 0, 255), 2)
    cv2.line(frame, (296, 256), (430, 265), (0, 0, 255), 2)
    
    # Find how find the line is from the car 
    center_top_line_x = (430 + 296) // 2
    center_top_line_y = (256 + 265) // 2
    center_right_line_x = (669 + 987) // 2
    center_right_line_y = (549 + 378) // 2

    # list used for knn and k value
    distance_top_line = []
    distance_right_line = []
    k = 1

    #Center a dot in the middle of the car and add visuals
    for info in car_id:
        # Get the coordinates
        x3, y3, x4, y4, id = info
        center_x = (x3 + x4) // 2
        center_y = (y3 + y4) // 2
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        # Perform the knn algo to find the closest car to the line
        # Find distance 
        ed_top_line = math.sqrt((center_top_line_x - center_x)**2 + (center_top_line_y - center_y)**2)
        ed_right_line = math.sqrt((center_right_line_x - center_x)**2 + (center_right_line_y - center_y)**2)
        distance_top_line.append((ed_top_line, (center_x, center_y)))
        distance_right_line.append((ed_right_line, (center_x, center_y)))

        #cv2.line(frame, (center_top_line_x, center_top_line_y), (center_x, center_y), (255, 0, 0), 2)
    
    #Find the closest to lines
    distance_top_line = sorted(distance_top_line)[:k]
    distance_right_line = sorted(distance_right_line)[:k]

    #Add lines
    car_top_line = distance_top_line[0]
    car_right_line = distance_right_line[0]
    closest_car_cooridnates_top_line = car_top_line[1]
    closest_car_cooridnates_right_line = car_right_line[1]
    closest_car_x_top_line, closest_car_y_top_line = closest_car_cooridnates_top_line
    closest_car_x_right_line, closest_car_y_right_line = closest_car_cooridnates_right_line
    cv2.line(frame, (closest_car_x_top_line, closest_car_y_top_line), (center_top_line_x, center_top_line_y), (255, 0, 0), 2)
    cv2.line(frame, (closest_car_x_right_line, closest_car_y_right_line), (center_right_line_x, center_right_line_y), (255, 0, 0), 2)

    return frame

#Function to display image
def displayVideo():
# Loop with all the logic
    while True:
        # Measure teh time it takes
        start_time = time.time()

        # Read all the frame 
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frames
        frame = processFrame(frame)
        cv2.imshow("video", frame)

        # print time
        end_time = time.time()
        time_change = end_time - start_time
        print("Time taken: {}".format(time_change))

        # Escape key to leave
        if cv2.waitKey(3) == 27:
            break

if __name__ == "__main__":

    # upload the trained classifier and video
    car_model = YOLO("yolov9c.pt")
    video = "output.mp4"
    cap = cv2.VideoCapture()
    cap.open(video)
    tracker = Tracker()

    # list of all objects model detects
    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Start thread
    # display_thread = threading.Thread(target=displayVideo)
    # display_thread.start()
    #display_thread.join()
    displayVideo()

    # Kill process
    cap.release()
    cv2.destroyAllWindows()





