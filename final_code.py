import cv2
from collections import OrderedDict
import numpy as np
import random
# Create a result with dimensions 1000x1000 pixels and three color channels (BGR)
bg_image = np.zeros((600, 700, 3), np.uint8)


all_rect_position_dict = {}
cv2.rectangle(bg_image,(100,100),(600,400),(255,0,47),5)


def check_point_position(x, lines_dict):
    """
    Check the position of a point within the line segments defined in a dictionary.

    Parameters:
    - x: Tuple (x, y) representing the point to check.
    - lines_dict: Dictionary with keys as positions and values as lists of endpoints.

    Returns:
    - List of positions where the point lies.
    """
    positions = []

    for position, line_endpoints in lines_dict.items():
        endpoint1, endpoint2 = line_endpoints
        x1, y1 = endpoint1
        x2, y2 = endpoint2

        # Check if x lies within the line segment
        if x1 <= x[0] <= x2 and min(y1, y2) <= x[1] <= max(y1, y2):
            positions.append(position)

    return positions[0]


def find_midpoint(points):
    
    point1 = points[0]
    point2 = points[1]

    mid_x = (point1[0] + point2[0]) // 2
    mid_y = (point1[1] + point2[1]) // 2
    return mid_x, mid_y


x1 = 100
y1 = 100
x2 = 200
y2 = 200
increase_node = 0
for i in range(0,15):
    cv2.rectangle(bg_image,pt1=(x1,y1),pt2=(x2,y2),color=(255,255,255),thickness=-1)
    cv2.rectangle(bg_image,pt1=(x1,y1),pt2=(x2,y2),color=(0,0,255),thickness=2)

    all_rect_position_dict[str(i+1)] = [(x1,y1),(x2,y2)]
    increase_node += 1
    x1 += 100
    x2 += 100
    if x1 >= 600:
        x1 = 100
        y1 += 100
    if x2 >= 700:
        x2 = 200
        y2 += 100 
    # print( str(i+1) ,(x1,y1),(x2,y2))



racks_availabity = {'rack_1': ['11', '6', '1'], 'rack_2': ['12', '7', '2'], 'rack_3': ['13', '8', '3'], 'rack_4': ['14', '9', '4'], 'rack_5': ['15', '10', '5']}

opponent_racks_availabity = {'rack_1': [], 'rack_2': [], 'rack_3': [], 'rack_4': [], 'rack_5': []}
my_racks_availabity = {'rack_1': [], 'rack_2': [], 'rack_3': [], 'rack_4': [], 'rack_5': []}
ball_filled_in_racks = OrderedDict(my_racks_availabity)   
max_ball_list = []
occupied_slots = []
count_to_stop = 0

# puting my ball first in rack

first_ball_select = random.choice(['rack_1', 'rack_2', 'rack_3', 'rack_4', 'rack_5'])
key_first_ball_select =  racks_availabity[first_ball_select][0]
circle_center = find_midpoint(all_rect_position_dict[key_first_ball_select])
cv2.circle(bg_image,center=circle_center,radius=40,color=(0,0,255),thickness=-1)

# adding my ball occupied in my_racks_availabity
my_racks_availabity[first_ball_select] = [(first_ball_select, key_first_ball_select)]

# removing ball from racks_availabity after it is add to rack
if key_first_ball_select not in occupied_slots:
    occupied_slots.append(key_first_ball_select)

    for rack, values in racks_availabity.items():
        if key_first_ball_select in values:
            values.remove(key_first_ball_select)
            

def click(event,x,y,flags,param):
    global racks_availabity,max_ball_list,count_to_stop
    reak_position = {'rack_1': [(100, 100), (200, 400)], 'rack_2': [(200, 100), (300, 400)], 'rack_3': [(300, 100), (400, 400)], 'rack_4': [(400, 100), (500, 400)], 'rack_5': [(500, 100), (600, 400)]}
    rack_clustered_position = {'rack_1': {'11': [(100, 300), (200, 400)], '6': [(100, 200), (200, 300)], '1': [(100, 100), (200, 200)]}, 'rack_2': {'12': [(200, 300), (300, 400)], '7': [(200, 200), (300, 300)], '2': [(200, 100), (300, 200)]}, 'rack_3': {'13': [(300, 300), (400, 400)], '8': [(300, 200), (400, 300)], '3': [(300, 100), (400, 200)]}, 'rack_4': {'14': [(400, 300), (500, 400)], '9': [(400, 200), (500, 300)], '4': [(400, 100), (500, 200)]}, 'rack_5': {'15': [(500, 300), (600, 400)], '10': [(500, 200), (600, 300)], '5': [(500, 100), (600, 200)]}}
    
    if event == cv2.EVENT_RBUTTONUP:
        
        if len(occupied_slots) != 15:
     
            keys_position = check_point_position(x=(x,y),lines_dict = reak_position)
            key_to_remove = racks_availabity[keys_position][0]
            
            # opponent ball slots adding
            for rack, values in opponent_racks_availabity.items():
                if keys_position == rack:
                    values.append((keys_position,key_to_remove))
            
            # removing ball from racks_availabity after it is add to rack
            if key_to_remove not in occupied_slots:
                occupied_slots.append(key_to_remove)
    
                for rack, values in racks_availabity.items():
                    if key_to_remove in values:
                        values.remove(key_to_remove)
                        
                circle_center = find_midpoint(all_rect_position_dict[key_to_remove])

                cv2.circle(bg_image,center=circle_center,radius=40,color=(255,0,0),thickness=cv2.FILLED)

            cv2.imshow("result",bg_image)

            
#-------------------------------------- logic for placing our ball -----------------------------------------
            
            max_ball_list = []
            for rack,val in racks_availabity.items():
                count_my_ball = len(my_racks_availabity[rack])
                count_opponent_ball = len(opponent_racks_availabity[rack])
                
                if count_my_ball == 1 and count_opponent_ball == 1:
                    sub_result = 2
                else:
                    sub_result = count_my_ball - count_opponent_ball
                print( my_racks_availabity[rack], "m",count_my_ball,'o',count_opponent_ball)
                max_ball_list.append((rack,sub_result))
                
            
                
            
            # Iterate through keys in a or b (assuming they have the same keys)
            for key in list(opponent_racks_availabity.keys()) + list(my_racks_availabity.keys()):
                # Concatenate the values for each key, handling cases where one dictionary may not have a key present
                
                opp =  opponent_racks_availabity.get(key, [])
                my = my_racks_availabity.get(key, [])
        
                if opp != [] or my != []:
                    opp = [(i[0],i[1],'blue') for i in opp ]
                    my = [(i[0],i[1],'red') for i in my ]

                ball_filled_in_racks[key] = my + opp
                

            for r,m in racks_availabity.copy().items():
                if m == []:
                    my_racks_availabity.pop(r)
                    opponent_racks_availabity.pop(r)
                    racks_availabity.pop(r)
                    max_ball_list = [ (i,j) for i,j in max_ball_list if i != r ]
            
         
            condition_check_first_3_ball = [ v for v,i in list(racks_availabity.items()) if len(i) == 3]            
            if count_to_stop < 3:
                
                keys_position = condition_check_first_3_ball[0]
                
                max_ball_list = []
                for rack,val in racks_availabity.items():
                    count_my_ball = len(my_racks_availabity[rack])
                    count_opponent_ball = len(opponent_racks_availabity[rack])
                    
                    if count_my_ball == 1 and count_opponent_ball == 1:
                        sub_result = 2
                        max_ball_list.append((rack,sub_result))
                
                        
                if max_ball_list == []:
                    keys_position = condition_check_first_3_ball[0]
                else:
                    keys_position = max(max_ball_list, key=lambda x: x[1])[0]
  
                    
                count_to_stop += 1
            else:
                if max_ball_list != []:
                    keys_position = max(max_ball_list, key=lambda x: x[1])[0]
           
            key_to_remove = racks_availabity[keys_position][0]
           
            
            # my ball slots adding
            for rack, values in my_racks_availabity.items():
                if keys_position == rack:
                    values.append((keys_position,key_to_remove))
            
            if key_to_remove not in occupied_slots:
                occupied_slots.append(key_to_remove)
    
                for rack, values in racks_availabity.items():
                    if key_to_remove in values:
                        values.remove(key_to_remove)
                        
                circle_center = find_midpoint(all_rect_position_dict[key_to_remove])
                cv2.circle(bg_image,center=circle_center,radius=40,color=(0,0,255),thickness=-1)
                
                
            print("ALL SLOTS ===========> ")

            print(my_racks_availabity)
   
            print("ALL SLOTS ===========> ")
            
            decision_take_dict = {}
            for i,j in ball_filled_in_racks.items():
                j.sort(key=lambda x: int(x[1]) , reverse=True) 
                decision_take_dict[i]  = j
                
            my_rack_count = 0
            for i,j in decision_take_dict.items():       
                if len(j) == 3:
                    if j[0][2] == 'red' and j[1][2] == 'red' and j[2][2] == 'red':
                        my_rack_count += 1
                    if j[0][2] == 'red' and j[1][2] == 'blue' and j[2][2] == 'red':
                        my_rack_count += 1
                    if j[0][2] == 'blue' and j[1][2] == 'red' and j[2][2] == 'red':
                        my_rack_count += 1
            
            print('!!!!!!!!!!!!!!',count_to_stop)            
                    
            if my_rack_count == 3:
                # Use cv2.putText() to write text on the image
                cv2.putText(bg_image, "YOU WON THE GAME", (160,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                print("----------------- YOU WON THE GAME ------------------------")
                        

        else:
            print("ALL SLOTS ARE OCCUPIED !!!!!!!")
        
            
        
        cv2.imshow("result",bg_image)
        

# Display the result in a window
cv2.namedWindow('result')
cv2.setMouseCallback("result",click)
cv2.imshow('result', bg_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
