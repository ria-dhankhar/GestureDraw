import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

######################
brushThickness = 15
eraserThickness = 100
drawColor = (255, 0, 255)
######################

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Read a test frame to get size
success, img = cap.read()
frame_height, frame_width = img.shape[:2]

# Resize the header to match the frame width
# Initial default brush and header
header = cv2.resize(overlayList[0], (frame_width, 125))
drawColor = (0, 0, 255)  # Default: Red

header_height = 125

# ✅ Helper function to detect which fingers are up
def fingersUp(lmList):
    fingers = []
    tipIds = [4, 8, 12, 16, 20]

    # Thumb
    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    # Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = fingersUp(lmList)

        # 4. Selection mode - 2 fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            print("Selection Mode")

            if y1 < 125:  # header area
                if 250 < x1 < 450:
                    header = cv2.resize(overlayList[0], (frame_width, 125))
                    drawColor = (0, 0, 255)  # Red
                elif 550 < x1 < 750:
                    header = cv2.resize(overlayList[1], (frame_width, 125))
                    drawColor = (0, 255, 0)  # Green
                elif 800 < x1 < 950:
                    header = cv2.resize(overlayList[2], (frame_width, 125))
                    drawColor = (255, 0, 255)  # Bright Purple
                elif 1050 < x1 < 1200:
                    header = cv2.resize(overlayList[3], (frame_width, 125))
                    drawColor = (0, 0, 0)  # Eraser (Black)

        # 5. Drawing mode - index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Merge the drawings with camera
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Overlay the header
    img[0:header_height, 0:frame_width] = header

    # Optional blending (you can remove if not needed)
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Virtual Painter", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inverted", imgInv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
