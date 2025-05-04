import cv2
import mediapipe as mp
import numpy as np
import time
import os
import hand_tracking_module as htm

## getting the iamge paths
folder_path = "header"
my_list = os.listdir(folder_path)
# print(my_list)

overlay_list = []
for img_list in my_list:
    image = cv2.imread(f'{folder_path}/{img_list}')
    overlay_list.append(image)
print(len(overlay_list))


cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
cap.set(3, 1280)
cap.set(4, 720)

time.sleep(2)  # Give the camera time to initialize

p_time = 0
c_time = 0

Header = overlay_list[0]    ## initializing the first image

draw_color = (255, 0, 255)       ## by default

detector = htm.hand_detector(detection_con=0.85)

## variables
brush_thickness = 15
eraser_thickness = 50
xp, yp = 0, 0

img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. import image
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture image")  
        break  # Exit if no frame is captured
    
    img = cv2.flip(img, 1)    ## flip image to avoid mirror effect
    
    
    #2. find hand landmarks
    img = detector.find_hands(img)
    lm_list, b_box = detector.find_position(img, draw = False)
    if len(lm_list) != 0:
        # print(lm_list)
        
        # tip of index and middle finger
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        
        
        #3. check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        
    
        #4. if selection mode - two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0                                     #####  reset the previous point
            print("Selection mode open")
            ## check the click
            if y1 < 150:
                if 300 < x1 < 500:
                    Header = overlay_list[0]
                    draw_color = (255, 0, 0)
                elif 600 < x1 < 800:
                    Header = overlay_list[1]
                    draw_color = (255, 255, 0)
                elif 800 < x1 < 1000:
                    Header = overlay_list[2]
                    draw_color = (0, 0, 255)
                elif 1000 < x1 < 1300:
                    Header = overlay_list[3]
                    draw_color = (0, 0, 0)
                    
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                    
    
        #5. Drawing mode - if index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, draw_color, cv2.FILLED)
            print("Drawing mode open")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1              ## from preventing to draw a line from the (o, o) to x1, y1
                
            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)     
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)      
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)      ## drawing in the webcam canvas
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)      ## drawing in the image canvas
            
            xp, yp = x1, y1
            
    
    #### merging the webcam canvas and draw canvas together
    img_gray  = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)                      ## check this part very carefully
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)          ##"_," means ignoring the return value
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)
    
    ###
    
    
    # Resize Header to match img width
    Header = cv2.resize(Header, (img.shape[1], 150))  
    img[0:150, 0:img.shape[1]] = Header  # Overlay header on top
    
    
    ######
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)      ## merging the window canvas and draw canvas, but it is not good enough, so we are trying another method above
    #############
    
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    
    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    
    cv2.imshow("Image", img)  
    # cv2.imshow("Image canvas", img_canvas)             ## it will give me the binary canvas
    # cv2.imshow("Image inverse  canvas", img_inv)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit