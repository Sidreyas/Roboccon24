
balls_pattern = {'rack_1': {'blue_ball':0,'red_ball':1}, 
                 'rack_2': {'blue_ball':0,'red_ball':1}, 
                 'rack_3': {'blue_ball':0,'red_ball':1}, 
                 'rack_4': {'blue_ball':2,'red_ball':0}, 
                 'rack_5': {'blue_ball':0,'red_ball':0}}

emtpy_racks = []
my_ball_check_count = 0

# finding empty racks
for rack,ball_count in balls_pattern.items():
    if ball_count['red_ball'] != 0:
        my_ball_check_count += 1
    if ball_count['blue_ball'] == 0 and ball_count['red_ball'] == 0:
        emtpy_racks.append(rack)

print(emtpy_racks, my_ball_check_count )

# removing filled in racks
for rack,ball_count in list(balls_pattern.items()):
    total_ball_count_in_rack = ball_count['blue_ball'] + ball_count['red_ball']
    if total_ball_count_in_rack == 3:

        balls_pattern.pop(rack)


max_ball_list = []
# rack finding logic
if len(emtpy_racks) == 0 or my_ball_check_count >= 3:
    
    print("FIRST LOOP entered in after 3 ball placed in rack !!!!!!!!!!!!!!!!")
    
    for rack,ball_count in balls_pattern.items():
        count_my_ball = ball_count['red_ball']
        count_opponent_ball = ball_count['blue_ball']
        
        if count_my_ball == 1 and count_opponent_ball == 1:
            sub_result = 2
        else:
            sub_result = count_my_ball - count_opponent_ball
        print(balls_pattern[rack], "m",count_my_ball,'o',count_opponent_ball)
        max_ball_list.append((rack,sub_result))
        
    
if my_ball_check_count < 3 :
    
    print("SECOND LOOP entered in before first 3 ball placed in rack !!!!!!!!!!!!!!!!")
    
    for rack,ball_count in balls_pattern.items():
        count_my_ball = ball_count['red_ball']
        count_opponent_ball = ball_count['blue_ball']
        
        if count_my_ball == 1 and count_opponent_ball == 1:
            sub_result = 2
            max_ball_list.append((rack,sub_result))
        
if max_ball_list == []:
    keys_position = emtpy_racks[0]
else:
    keys_position = max(max_ball_list, key=lambda x: x[1])[0]
    
print( "result_rack ===> ", keys_position)