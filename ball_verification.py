
#!/usr/bin/env python3

import cv2
import rospy
from std_msgs.msg import Int32
from ultralytics import YOLO
import os


os.environ['QT_QPA_PLATFORM'] = 'xcb'


def main():
    # Initialize the ROS node
    rospy.init_node('yolo_ball_detector', anonymous=True)
    ball_colour_pub = rospy.Publisher('colour_ball', Int32, queue_size=10)
    rate = rospy.Rate(10)


    model = YOLO(r"model_long_ball.pt")  


    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        frame = frame[10:600,10:600]

        if not ret:
            rospy.loginfo("Failed to grab frame")
            break


        results = model.predict(frame,conf=0.60,verbose = True)

        if results and len(results) > 0:
            detections = results[0].boxes 
            print("*****B4 for loop*****") 
            if detections:
                for detection in detections:
                    x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
                    conf = detection.conf[0]
                    cls = int(detection.cls[0])
                    rospy.loginfo(f"Detected class: {cls}")
                    print("*****CLS*****", cls)
                    ball_colour_pub.publish(cls)
                    
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cls =10
                ball_colour_pub.publish(cls)
                print("No detections ===>", cls)

        cv2.imshow('YOLOv8 Object Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
