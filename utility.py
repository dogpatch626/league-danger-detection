import numpy as np
import cv2
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
    # cv2.imshow(mask)
    print(red_proportion)

    return red_proportion >= red_threshold
    

def isolate_top_bottom(frame, side_height):
    height, width, _ = frame.shape

    # Top side
    top_side = frame[:side_height, :]

    # Bottom side
    # bottom_side = frame[height - side_height:, :]

    return top_side

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



