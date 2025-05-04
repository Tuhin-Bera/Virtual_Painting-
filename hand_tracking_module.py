import cv2
import mediapipe as mp
import time
import math



class hand_detector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con
         
        self.mp_hands = mp.solutions.hands                                                                       # Loads MediaPipe's hand tracking solution
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, 
                                         max_num_hands=self.max_hands, 
                                         min_detection_confidence=float(self.detection_con), 
                                         min_tracking_confidence=float(self.tracking_con))                 # create a hand detection model instance
        self.mpdraw = mp.solutions.drawing_utils                                                                  # it helps to draw the hand points
        self.tip_ids = [4, 8, 12, 16, 20]

    

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert the img BGR to RGB
        self.results = self.hands.process(imgRGB)      # proccesses the image to detect hands
    
    
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
        return img
        
    
    
    
    def find_position(self, img, hand_no = 0, draw=True):
        x_list = []
        y_list = []
        b_box = (0, 0, 0, 0)
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                                
                                h, w, c = img.shape           
                                cx, cy = int(lm.x*w), int(lm.y*h)
                                x_list.append(cx)
                                y_list.append(cy)
                                # print(id, cx, cy)  
                                self.lm_list.append([id, cx, cy]) 
                                if draw:  
                                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if x_list and y_list:  # only if both lists are not empty
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            b_box = x_min, y_min, x_max, y_max


            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                (0, 255, 0), 2)
                                        
        return self.lm_list, b_box
    

    
    
    def fingers_up(self):
        fingers = []
        
        if len(self.lm_list) == 0:
            return fingers
        
        if len(self.lm_list) != 0:
            ## this is for the thumb finger  ( Right Hand )
            if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            ## this will work for the four fingers except thumb finger
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # totalFingers = fingers.count(1)q
            # print(fingers)
        return fingers
    
    
    def find_distance(self, p1, p2, img, draw=True,r=10, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
        
                    

def main():
    p_time = 0  # previous time
    c_time = 0  # current time

    cap = cv2.VideoCapture(0)  # Try different indices if needed (0, 1, 2, ......)

    # if not cap.isOpened():
    #     print("Error: Could not open camera.")
    #     exit()

    # time.sleep(2)  # Give the camera time to initialize

    detector = hand_detector()  


    while True:
        success, img = cap.read()

        if not success or img is None:  # ✅ Added back error checking
            print("Error: Failed to capture image")
            break




        img = detector.find_hands(img)  # ✅ Call find_hands on the image
        lm_list, b_box = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])



        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)

        cv2.imshow("Image", img)  

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break  # Press 'q' to exit





if __name__ == "__main__":
    main()

