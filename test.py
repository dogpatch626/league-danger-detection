import cv2


capture = cv2.VideoCapture(1)

while(True):
    ret, frame = capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.imshow('Virtual Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

