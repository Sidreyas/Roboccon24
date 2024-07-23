import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from time import sleep
from motor_control.srv import OneInt, OneIntRequest, OneIntColour, OneIntColourRequest
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

class CustomYOLO(YOLO):
    def set_class_names(self, names):
        self._names = names

    @property
    def names(self):
        return self._names

model = CustomYOLO(r"model_long_ball.pt", task='detect')

class_names = {"0": "blue", "1": "purple", "2": "red", "3": "silo"}

model.set_class_names(class_names)

def call_motor_control_services_silo():
    rospy.wait_for_service('send_silo_drop')
    try:
        send_silo_drop = rospy.ServiceProxy('send_silo_drop', OneInt)
        int_request = OneIntRequest(a=1)
        int_response = send_silo_drop(int_request)
        rospy.loginfo("Service call to send_silo_drop successful: %s", int_response.success)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def switch_camera(pipeline, config, new_device_serial):
    try:
        # pipeline.stop()
        config.enable_device(new_device_serial)
        pipeline.start(config)
        return rs.align(rs.stream.color)
    except Exception as e:
        rospy.logerr("Error while switching camera: %s", e)
        return None

def get_camera(new_device_serial):
    pipeline = rs.pipeline()
    config = rs.config()
    W = 640
    H = 480
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_device(new_device_serial)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline,align

def point_inside(rectangle, entire_coordinates):
    x1, y1, x2, y2, _, _ = rectangle
    return (entire_coordinates[0] <= x1 <= entire_coordinates[2] and entire_coordinates[1] <= y1 <= entire_coordinates[3]) or (entire_coordinates[0] <= x2 <= entire_coordinates[2] and entire_coordinates[1] <= y2 <= entire_coordinates[3])

def get_rack_decision(balls_pattern, our_ball):
    ball_categories = ['blue_ball', 'red_ball']

    my_ball_colour = ball_categories[our_ball]
    opponent_ball_colour = ball_categories[1 - our_ball]

    empty_racks = []
    my_ball_check_count = 0

    for rack, ball_count in balls_pattern.items():
        if ball_count[my_ball_colour] != 0:
            my_ball_check_count += 1
        if ball_count[my_ball_colour] == 0 and ball_count[opponent_ball_colour] == 0:
            empty_racks.append(rack)

    max_ball_list = []
    if len(empty_racks) == 0 or my_ball_check_count >= 3:
        for rack, ball_count in balls_pattern.items():
            count_my_ball = ball_count[my_ball_colour]
            count_opponent_ball = ball_count[opponent_ball_colour]

            if count_my_ball == 1 and count_opponent_ball == 1:
                sub_result = 2
            else:
                sub_result = count_my_ball - count_opponent_ball

            max_ball_list.append((rack, sub_result))

    if my_ball_check_count < 3:
        for rack, ball_count in balls_pattern.items():
            count_my_ball = ball_count[my_ball_colour]
            count_opponent_ball = ball_count[opponent_ball_colour]

            if count_my_ball == 1 and count_opponent_ball == 1:
                sub_result = 2
                max_ball_list.append((rack, sub_result))


    if not max_ball_list and empty_racks:
        keys_position = empty_racks[0]
    elif max_ball_list:
        keys_position = max(max_ball_list, key=lambda x: x[1])[0]
    else:
        keys_position = "0"  

    print("Selected Rack:", keys_position)
    return keys_position

def decision_maker(image_input_def, depth_frame,depth_image, our_ball, last_known_silos):
    global model

    balls_pattern = {'rack_1': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_2': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_3': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_4': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_5': {'blue_ball': 0, 'red_ball': 0}}
    
    input_image = image_input_def
    results = model(input_image, conf=0.35,verbose = False)

    balls_detected_coordinates = list(filter(lambda x: x[-1] in [0, 1, 2], results[0].boxes.cpu().numpy().data.tolist()))
    detected_silo_coordinates = sorted(list(filter(lambda x: x[-1] == 3.0, results[0].boxes.cpu().numpy().data.tolist())), key=lambda x: x[0])
    
    for i in range(5 - len(detected_silo_coordinates)):
        detected_silo_coordinates.append([]) 

    print("detected_silo_coordinates --> ",detected_silo_coordinates)

    if detected_silo_coordinates != [[], [], [], [], []]:
        finded_balls_in_silo = {}
        
        for id, i in enumerate(detected_silo_coordinates):
            l = []
            d = {}
            if i != []:
                for j in balls_detected_coordinates:
                    res_inside = point_inside(j, i)
                    if res_inside:
                        l.append((j[-1]))
                blue_ball_count = l.count(0.0)
                red_ball_count = l.count(2.0)
                d['blue_ball'] = blue_ball_count
                d['red_ball'] = red_ball_count
                finded_balls_in_silo["rack_" + str(id + 1)] = d
            else:
                d['blue_ball'] = -6
                d['red_ball'] = -3
                finded_balls_in_silo["rack_" + str(id + 1)] = d

        for key in balls_pattern.keys():
            balls_pattern[key] = finded_balls_in_silo.get(key, balls_pattern[key])

        last_known_silos = detected_silo_coordinates
        keys_position = get_rack_decision(balls_pattern, our_ball)

    else:
        keys_position = "0" 
        


    final_image_result = results[0].plot()
    final_image_result = cv2.putText(final_image_result, str(keys_position), (450, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1)
    
    object_coordinates = []

    if keys_position != "0":
        rack_index = int(keys_position.split('_')[1]) - 1
        if rack_index < len(last_known_silos):
            rack_coords = last_known_silos[rack_index]
            _, y1, _, y2, _, _ = rack_coords
            x = int((rack_coords[0] + rack_coords[2]) / 2)
            y = int((rack_coords[1] + rack_coords[3]) / 2)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            dept=depth_image[int(y),int(x)]/1000
            object_pos = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dept)
            if object_pos[2] > 0:
                object_coordinates.append(object_pos)
                final_image_result = cv2.circle(final_image_result, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        
    else:
        object_coordinates = []

    return keys_position, object_coordinates, final_image_result, last_known_silos

def run_yolo():
    def ball_clrCallback(msg):
        global ball_verification
        ball_verification = msg.data

    def d455Callback(msg):
        global change_to_d455
        change_to_d455 = msg.data

    def ai_enableCallback(msg):
        global ai_enable
        ai_enable = 0
        ai_enable = msg.data

    rospy.init_node('robot_mover', anonymous=True)

    global ai_enable, change_to_d455, ball_verification

    ai_enable = rospy.Subscriber("ai_enable_action", Int32, ai_enableCallback)
    ball_verification = rospy.Subscriber("colour_ball", Int32, ball_clrCallback)
    d455_sub = rospy.Subscriber("switch_to_d455", Int32, d455Callback)


    global team_clr_ball
    W = 640
    H = 480

    constant_distance = 5
    first_split_d = constant_distance / 3
    first_split_d_l = constant_distance / 2
    total_distance_to_cover_for_angular = [0 , first_split_d , first_split_d * 2, first_split_d * 3] 
    total_distance_to_cover_for_linear = [0 , first_split_d_l , first_split_d_l * 2]
    
    switch_camera_ball_pick = True
    switch_camera_silo = False

    d455_serial = "234322307142"
    d435_serial = "834412070087"

    pipeline_ball , ball_camera_align = get_camera(d435_serial)
    pipeline_silo , silo_camera_align = get_camera(d455_serial)

    # config = rs.config()
    # config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    # config.enable_device(d435_serial)

    # pipeline = rs.pipeline()
    # profile = pipeline.start(config)

    # align = rs.align(rs.stream.color)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    team_clr_ball = 2

    our_ball = 1  # Set this to 0 for blue ball, 1 for red ball

    not_ball_detected_count = 0
    last_known_silos = []
    flag_for_silo_turn = False

    while not rospy.is_shutdown():

        if switch_camera_ball_pick:
            frames = pipeline_ball.wait_for_frames()
            aligned_frames = ball_camera_align.process(frames)
        if switch_camera_silo :
            frames = pipeline_silo.wait_for_frames()
            aligned_frames = silo_camera_align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        display_image = color_image
        

        if switch_camera_ball_pick and ai_enable == 1:

            results = model.predict(color_image, conf=0.25,verbose = False)

            object_coordinates = []
            for r in results:
                boxes = r.boxes  
           
                for box in boxes:
                    
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()
                    c = int(box.cls)
                   
                    if class_names.get(str(c)) not in ["red", "silo"]:
                        continue
                
                    cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                    x = (int(b[0]) + int(b[2])) / 2
                    y = (int(b[1]) + int(b[3])) / 2
                    dept = depth_image[int(y), int(x)] / 1000
                    if dept > 2.0:
                        dept += 0.6
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    object_pos = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dept)
                    if object_pos[2] > 0:
                        object_coordinates.append((object_pos, class_names.get(str(c))))
         
            if object_coordinates :

                not_ball_detected_count = 0
                
                min_z_coordinate, obj_class = min(object_coordinates, key=lambda coord: coord[0][2])
                distance_Z = min_z_coordinate[2]
                
                print("Change to d455:",change_to_d455)
                if change_to_d455 == 1 and ball_verification == team_clr_ball:                      
                    # align = switch_camera(pipeline, config, d455_serial)
                    move_cmd = Twist()
                    move_cmd.linear.x = 0.0
                    move_cmd.angular.z = 0.0
                    pub.publish(move_cmd)

                    print("Destination (Blue Ball) reached. Stopping the robot in " , distance_Z,'coordinates')

                    print("*****Switching to D455*****")
                    switch_camera_ball_pick = False
                    switch_camera_silo = True
                    not_ball_detected_count = 0

                else:

                    if obj_class == "red":

                        move_cmd = Twist()

                        if total_distance_to_cover_for_linear[1] < distance_Z <= total_distance_to_cover_for_linear[2]:
                            move_cmd.linear.x = 1.0
                        elif total_distance_to_cover_for_linear[0] < distance_Z <= total_distance_to_cover_for_linear[1]:
                            move_cmd.linear.x = 0.55
                        if distance_Z >= total_distance_to_cover_for_linear[2] : 
                            move_cmd.linear.x = 1.0
        

                        if total_distance_to_cover_for_angular[2] < distance_Z <= total_distance_to_cover_for_angular[3]:
                            move_cmd.angular.z = -0.18 * min_z_coordinate[0]
                            print("first")
                        elif total_distance_to_cover_for_angular[1] < distance_Z <= total_distance_to_cover_for_angular[2]:
                            move_cmd.angular.z = -0.58 * min_z_coordinate[0]
                            print("second")
                        elif total_distance_to_cover_for_angular[0] < distance_Z <= total_distance_to_cover_for_angular[1]:
                            move_cmd.angular.z = -0.85 * min_z_coordinate[0]
                            print("thrid")
                        if distance_Z >= total_distance_to_cover_for_angular[3] : 
                            print("not any oneee")
                            move_cmd.angular.z = -0.18 * min_z_coordinate[0]

                        pub.publish(move_cmd)

                    else:
                        print("Red ball not detected..")
                          
            else:

                move_cmd = Twist()

                print('Nothing Detected')
                
                not_ball_detected_count += 1
                if not_ball_detected_count > 40:
                    move_cmd.linear.x = - 0.5
                    move_cmd.angular.z = - 0.8
                    

                pub.publish(move_cmd)

       
            # cv2.imshow('FRONT CAMERA RESULTS',results[0].plot())
            display_image = results[0].plot()


        if switch_camera_silo and ai_enable == 1:

            position, object_coordinates, result_image, last_known_silos = decision_maker(color_image, depth_frame,depth_image, our_ball, last_known_silos)

            move_cmd = Twist()
            print('object_coordinates ---> ',object_coordinates)
            if object_coordinates:

                flag_for_silo_turn = True
                object_x = object_coordinates[0][0]
                print('distance -->',object_coordinates[0][2])
                print('X-coordinates-->',object_coordinates[0][0])


                if object_coordinates[0][2] > 1.8:
                    move_cmd.linear.x = -1.2
                    move_cmd.angular.z = -0.44 * object_x
                    print('first')

                if 1.5 < object_coordinates[0][2] <=1.9:

                    if object_coordinates[0][0] > 0:
                        move_cmd.linear.y = 0.5
                        if -0.04 <= object_coordinates[0][0] <= 0.04 :
                            move_cmd.linear.x = - 0.7

                    elif object_coordinates[0][0] < 0:
                        move_cmd.linear.y = - 0.5
                        if -0.04 <= object_coordinates[0][0] <= 0.04 :
                            move_cmd.linear.x = - 0.7

                    # if -0.04 <= object_coordinates[0][0] <= 0.04 :
                    #     print("Y-Direction")
                    #     move_cmd.linear.x = - 0.7
                    #     move_cmd.angular.z = - 0.3 * object_x
                    # else:
                    #     if object_coordinates[0][0] > 0:
                    #         move_cmd.linear.y = 0.5
                    #     if object_coordinates[0][0] < 0:
                    #         move_cmd.linear.y = - 0.5

                    print('second')

                else:

                    if object_coordinates[0][2] < 0.36 :
                        move_cmd.linear.x = 0.0
                        move_cmd.angular.z = 0.0
                        print("destination reached in .................................." , object_coordinates[0][2] )

                        # align = switch_camera(pipeline, config, d435_serial)
                        call_motor_control_services_silo()

                        sleep(2)
                        print("******")
                        print('SILO Service Called')
                        print("*****")

                        switch_camera_ball_pick = True
                        switch_camera_silo = False
                        flag_for_silo_turn = False

                        
                        print("*****Switching to D435*****")

                    else:

                        print('else la else da ....')
                        
                        move_cmd.linear.x = - 0.55

            else:

                if not flag_for_silo_turn:
                    move_cmd.linear.x = 0.0
                    move_cmd.angular.z = - 0.6
                

                
            pub.publish(move_cmd)

            # cv2.imshow('REAR CAMERA RESULTS', result_image)
            display_image = result_image
        
        
        cv2.imshow('result', cv2.resize(display_image,(640,480)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            move_cmd = Twist()
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0
            pub.publish(move_cmd)
            break

    move_cmd = Twist()
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    pub.publish(move_cmd)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("func called")
    run_yolo()
    print("bye")
