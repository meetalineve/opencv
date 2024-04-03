import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv hue sat value
    lower_red = np.array([130, 100, 100])
    upper_red = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding box around the largest contour (assuming the ball is the largest object)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area of the largest contour
        area = cv2.contourArea(largest_contour)
        
        # Set a threshold for minimum area to consider it as the whole ball
        min_area_threshold = 45000  # Adjust as needed
        
        if area > min_area_threshold:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw center
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
            
            # Print coordinates of the center
            print("Center coordinates: ({}, {})".format(center_x, center_y))
    
    # Display the frames
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
