# vid id in the cv2.capture formmat only 

import cv2   
import mediapipe as mp 
import time 
 
 
class HandDetector() : 
    def __init__(self, mode=False, maxHands = 2 , detectionCon = 0.5, trackCon = 0.5):
        #these are the og, parameters of the HANDS object... 
        self.mode = mode 
        self.maxHands = maxHands 
        self.detectionCon = detectionCon 
        self.trackCon = trackCon
        
        # Objects needed for this are hand object and the Drawing Object...
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.maxHands,
                model_complexity=1,  # Default complexity
                min_detection_confidence=float(self.detectionCon),  # Ensure float
                min_tracking_confidence=float(self.trackCon)  # Ensure float
        )
        self.mpDrawObj = mp.solutions.drawing_utils
    
    def handFinds(self, frame, draw=True): 
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        self.res = self.hands.process(frameRGB) #process the HANDS IN THE IMAGES 
         
        if self.res.multi_hand_landmarks : #DRAEING THE HANDS....
            # for every hand in frame 
            for lmkr in self.res.multi_hand_landmarks: 
                if draw: 
                   self.mpDrawObj.draw_landmarks(frame, 
                                                 lmkr, 
                                                 self.mpHands.HAND_CONNECTIONS)
        return frame 
    
    def findPosition(self, frame, handNo=0, draw=True): 
        # for every point on the LandMark # denoted by the ID 
        landMark = [] # LANDMARK LIST 
        if self.res.multi_hand_landmarks : 
          myHand = self.res.multi_hand_landmarks[handNo] 
          
          for id, lm in enumerate(myHand.landmark) :
             h, w, c = frame.shape 
             cx, cy = int(lm.x * w) , int(lm.y * h) # Getting thy coordinates 
             # print(f"id: {id} | coordinates: ({cx} ,{cy})")
             
             landMark.append([id, cx, cy]) 
             if draw : 
                cv2.circle(frame, (cx,cy), 10, (0,225,0), cv2.FILLED)
                
        return landMark

def main() : 
    ctime = 0 
    ptime = 0 
    vid = cv2.VideoCapture(1)
    detector = HandDetector() # DETECTOR OBJECT FOR THE HAND DETECTION... 
    
    while True : 
        ret, frame = vid.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        
        
        
        frame = detector.handFinds(frame=frame) 
        lmlist = detector.findPosition(frame=frame)
        
        if len(lmlist) != 0 : 
           print(lmlist[0])  #palm position 
        
        
        ctime = time.time() 
        fps = 1/(ctime-ptime)
        ptime = ctime 
        # put frames rate on the screen  
        
        cv2.putText(img=frame,
                    text=(str(int(fps))),
                    org=(30, 100), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=3, 
                    color=(0, 255, 0))
        
        cv2.imshow('frame', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    vid.release()   
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__" : 
    main() 