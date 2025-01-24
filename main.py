import numpy as np 
import cv2 
import mediapipe as mp 
import time 
import handTracking_module as htm 
import math 

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# from handTracking import trackHands
vid = cv2.VideoCapture(1) 
ctime = 0 
ptime = 0 

detector = htm.HandDetector(detectionCon=0.7) # makeing the handdetector object 
 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)  

volRange = volume.GetVolumeRange() # range = (-63.5, 0.0, 0.5) 

# see notebook for better understanding 
minVol = volRange[0] 
maxVol = volRange[1]  
 

 

while True : 
    ret, frame = vid.read() 
    
    frame = detector.handFinds( frame=frame )
    lmList = detector.findPosition( frame=frame , draw=False)   
    
    # You actually found the points of index and the thumb 
    
    if len(lmList) != 0 : 
        # print(lmList[8], lmList 
        # getting coordinated of the index and thumb for the vol control
        
        x1, y1 = lmList[8][1] ,lmList[8][2] 
        x2, y2 = lmList[4][1] ,lmList[4][2] 
        
        cx, cy = int((x1+x2) / 2) , int((y1+y2) / 2) 
        
        cv2.circle(frame , (x1, y1), 10, (0, 255, 255), cv2.FILLED) 
        cv2.circle(frame , (x2, y2), 10, (0, 255, 255), cv2.FILLED)
        
        #color of the Midle of the Line 
        cv2.circle(frame , (cx, cy), 5, (0, 255, 255), cv2.FILLED) 
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255))
        
        length = math.hypot(x1-x2 , y1-y2) 
        
        # handRange vs Voloume range  
        # (45, 300)  and (-65, 0) 
        # Convert the Hand Range to Voloume Range 
        
        vol = np.interp(length, [25,160], [minVol, maxVol])
        actualVol = np.interp(vol , [minVol, maxVol], [0,100]) 
          
        print(length, vol) 
        volume.SetMasterVolumeLevel(vol, None) 
        cv2.putText(img=frame,
                    text=f"Volume: {int(actualVol)}%",
                    org=(30, 100), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, 
                    color=(0, 255, 0)
                    )
        
        if length<30: 
           cv2.circle(frame , (cx, cy), 10, (255, 0, 0), cv2.FILLED) 
            
    cv2.imshow('frame', frame)  
    
    
    # Frames output => 
    # ctime = time.time() 
    # fps = 1/(ctime-ptime)
    # ptime = ctime 
    # # put frames rate on the screen   
    # cv2.putText(img=frame,
    #             text=(str(int(fps))),
    #             org=(30, 100), 
    #             fontFace=cv2.FONT_HERSHEY_PLAIN,
    #             fontScale=3, 
    #             color=(0, 255, 0)
    # )
 
    if cv2.waitKey(1) & 0xFF == ord('q') :
      print(f"Video Exited!")     
      vid.release()   
      cv2.destroyAllWindows() 