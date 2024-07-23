from ultralytics import YOLO
import cv2

model = YOLO(r"/home/sudharsan/Documents/PROJECTS/Roboccon/Roboccon24/Final-Model.onnx")


# if hasattr(model, 'names'):
#     print("Model classes:")
#     for idx, class_name in model.names.items():
#         print(f"{idx}: {class_name}")
# else:
#     print("Model classes not found.")

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

    for rack, ball_count in list(balls_pattern.items()):
        total_ball_count_in_rack = ball_count[my_ball_colour] + ball_count[opponent_ball_colour]
        if total_ball_count_in_rack == 3:
            balls_pattern.pop(rack)

    max_ball_list = []
    # Rack finding logic
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
            
    if max_ball_list == []:
        keys_position = empty_racks[0]
    else:
        keys_position = max(max_ball_list, key=lambda x: x[1])[0]
        
    return keys_position

# Main function to make decisions based on the input image
def decision_maker(image_input_def, our_ball):
    global model

    # Initial balls pattern
    balls_pattern = {'rack_1': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_2': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_3': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_4': {'blue_ball': 0, 'red_ball': 0}, 
                     'rack_5': {'blue_ball': 0, 'red_ball': 0}}
    
    input_image = image_input_def
    
    # Model prediction
    results = model.predict(input_image, conf=0.3)

    balls_detected_coordinates = list(filter(lambda x: x[-1] in [0, 1, 2, 4], results[0].boxes.numpy().data.tolist()))
    detected_silo_coordinates = sorted(list(filter(lambda x: x[-1] == 3.0, results[0].boxes.numpy().data.tolist())), key=lambda x: x[0])

    if len(detected_silo_coordinates) == 5: 
        finded_balls_in_silo = {}
        
        for id, i in enumerate(detected_silo_coordinates):
            l = []
            d = {}
            for j in balls_detected_coordinates:
                res_inside = point_inside(j, i)
                if res_inside:
                    l.append((j[-1]))
            blue_ball_count = l.count(0.0)
            red_ball_count = l.count(2.0)
            d['blue_ball'] = blue_ball_count
            d['red_ball'] = red_ball_count
            finded_balls_in_silo["rack_" + str(id + 1)] = d
            
        balls_pattern['rack_1'] = finded_balls_in_silo['rack_1']
        balls_pattern['rack_2'] = finded_balls_in_silo['rack_2']
        balls_pattern['rack_3'] = finded_balls_in_silo['rack_3']
        balls_pattern['rack_4'] = finded_balls_in_silo['rack_4']
        balls_pattern['rack_5'] = finded_balls_in_silo['rack_5']
        
        keys_position = get_rack_decision(balls_pattern, our_ball)
    else:
        keys_position = 0        
        
    final_image_result  = results[0].plot()
    final_image_result = cv2.putText(final_image_result, str(keys_position), (450, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1)
    return keys_position, final_image_result

def set_team_color():
    while True:
        team_color = input("Enter your team color (blue or red): ").strip().lower()
        if team_color in ['blue', 'red']:
            return 0 if team_color == 'blue' else 2
        else:
            print("Invalid input. Please enter 'blue' or 'red'.")

def main():
    cap = cv2.VideoCapture(2)

    our_ball = set_team_color()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break

        position, result_image = decision_maker(frame, our_ball)

        cv2.imshow('Result', result_image)
        print("Position",position)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

