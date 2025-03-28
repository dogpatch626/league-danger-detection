
import numpy as np
import cv2
from playsound import playsound
import datetime

threshold = -0.35
def isolate_red (frame, red_threshold):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # red color boundary
        # Define red hue ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(frame, lower_red2, upper_red2)    
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    red_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    # Calculate proportion
    red_proportion = red_pixels / total_pixels
    
    print(red_proportion)

    return red_proportion >= red_threshold
    
    

    

lower_red = np.array([100, 20, 20], dtype = np.uint8)
upper_red = np.array([255, 100 , 100], dtype = np.uint8)

def isolate_top_bottom(frame, side_height):
    height, width, _ = frame.shape

    # Top side
    top_side = frame[:side_height, :]

    # Bottom side
    bottom_side = frame[height - side_height:, :]

    return top_side, bottom_side

def isolate_sides_mask(frame, side_width):
    height, width, _ = frame.shape

    # Create masks
    left_mask = np.zeros_like(frame)
    left_mask[:, :side_width] = 255

    right_mask = np.zeros_like(frame)
    right_mask[:, width - side_width:] = 255

    # Apply masks
    left_side = cv2.bitwise_and(frame, left_mask)
    right_side = cv2.bitwise_and(frame, right_mask)

    return left_side, right_side


def check_debounce(debounce, timestamp):
    
    delta = datetime.timedelta(minutes=4)
    if len(debounce) == 0:
        return True
    if timestamp - debounce[0] >= delta:
        return True
    return False

def main():
    template = cv2.imread('./smaller_warning.png' )
    template_mask = cv2.inRange(template, lower_red, upper_red)
    template = cv2.bitwise_and(template, template, mask = template_mask)
    
    debounce = []
    
    # captured_frames = cv2.VideoCapture('./league.mp4')
    captured_frames = cv2.VideoCapture(2)  
    backsub = cv2.createBackgroundSubtractorMOG2()
    while True:

        frame_bool, frame = captured_frames.read()

        timestamp = datetime.datetime.now()
        

        if frame is None:
            print('error reading frame')
            break
        frame = frame.astype(np.uint8)
        
        # resize to a smaller resolution so that we can process the frames faster
        downsized = cv2.resize(frame, (768, 364))
        top, bottom  = isolate_top_bottom(downsized, 30)
        is_red = isolate_red(top, red_threshold=0.030)
        print(is_red) 
        if is_red == True and check_debounce(debounce, timestamp):
            if len(debounce) == 0:
                print("damage Detected")
                debounce.append(timestamp)
                playsound("metal-pipe-clang.mp3")
            else:
                debounce.pop()  

    # Display the resulting frame
        cv2.imshow('Detected Objects',top)
        cv2.waitKey(1)
    # Break the loop if 'q' is pressed
        if 0xFF == ord('q'):
            break
    captured_frames.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
