import ScreenCaptureKit as ScKit
import numpy as np
import cv2
from playsound import playsound
import datetime
from utility import (isolate_red, isolate_sides_mask, isolate_top_bottom)
import time
from mss.darwin import MSS as mss


threshold = -0.35
    
lower_red = np.array([100, 20, 20], dtype = np.uint8)
upper_red = np.array([255, 100 , 100], dtype = np.uint8)


def rate_limit(last_used, timestamp):
    
    delta = datetime.timedelta(minutes=4)
    if len(last_used) == 0:
        return True
    if timestamp - last_used[0] >= delta.total_seconds():
        return True
    return False



with mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 1000, "left": 0, "width": 100, "height": 50}
    print(sct.monitors)
    template = cv2.imread('./smaller_warning.png' )
    template_mask = cv2.inRange(template, lower_red, upper_red)
    template = cv2.bitwise_and(template, template, mask = template_mask)
    
    last_used = []
    
    # captured_frames = cv2.VideoCapture(2)  

    while "Screen capturing":
        last_time = time.time()
        
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
                
        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)

        
        top  = isolate_top_bottom(img, 30)
        left, right = isolate_sides_mask(top, 30 )
        is_red = isolate_red(top, red_threshold=0.10)
        print(is_red) 
        if is_red == True and rate_limit(last_used, last_time):
            if len(last_used) == 0:
                print("damage Detected")
                last_used.append(last_time)
                # I'm sure you get the point
                # playsound("metal-pipe-clang.mp3")
            else:
                last_used.pop()  

        

        print(f"fps: {1 / (time.time() - last_time)}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break



