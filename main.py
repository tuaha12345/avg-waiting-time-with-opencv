import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
previous_time = time.time()
total_time_spent = 0
# Initialize variables
model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture('video.mp4')

# Load class labels
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")



# Video dimensions
video_width = 1020
video_height = 500
half_width = int(video_width / 2)
half_height = int(video_height / 2)

count = 0
count_time=0
not_in_position=0
bool_person=False
count_person=0
total_time_spent = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Process every 3rd frame to reduce computation
        continue

    frame = cv2.resize(frame, (video_width, video_height))
    results = model.predict(frame)
    boxes = results[0].boxes.boxes

    # Convert results to a Pandas DataFrame
    px = pd.DataFrame(boxes).astype("float")

    if len(boxes) > 0:
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            _id = int(row[5])
            class_name = class_list[_id]

            if class_name == 'person':
                #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                #cv2.putText(frame, f"***", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 255, 90), 2)

                # Check if the person is within the target region
                if ((half_width - 70 < x1 < 620) and y2>200) :
                    current_time = time.time()
                    elapsed_time = current_time - previous_time  # Calculate actual time elapsed
                    total_time_spent += elapsed_time
                    previous_time = current_time
                    total_time_spent=total_time_spent


                    #total_time_spent += 1 / frame_rate  # Increment total time spent
                    cv2.putText(frame, f"person:{count_person+1}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 250, 100), 1)
                    count_time += 1
                    print('count_time:',count_time)
                    if(count_time>10 and count_time<30):
                        
                        print('count_person:',count_person)
                        not_in_position=0
                        bool_person=True
                    elif(count_time>25):
                        count_time=0

                else:
                    seconds=0
                    not_in_position += 1
                    #print('not_in_position:',not_in_position)
                    if(not_in_position>150):
                        if(bool_person==True):
                            count_person += 1
                            bool_person=False


    # Display person count and total time spent on the screen
    
    cv2.putText(frame, f"People in place: {count_person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    show_total_time_spent = total_time_spent/10
    cv2.putText(frame, f"Total time: {show_total_time_spent:.3f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    show_avg_time_spent = show_total_time_spent / (count_person+1)
    cv2.putText(frame, f"Average spent time: {show_avg_time_spent:.2f}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 255), 2)

    # Draw target region lines
    cv2.line(frame, (450, 255), (650, 255), (10, 20, 120), 3)
    cv2.line(frame, (515, 180), (650, 180), (20, 120, 20), 3)
    cv2.line(frame, (half_width, 0), (half_width, video_height), (0, 250, 0), 3)
    cv2.line(frame, (0, half_height), (video_width, half_height), (0, 250, 0), 3)
    

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
