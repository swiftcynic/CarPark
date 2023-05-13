import cv2
import pickle
import cvzone
import numpy as np
import pandas as pd
import datetime

img = cv2.imread('carParkImg.png')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

empty =  [False for _ in range(len(posList))]

def checkParkingSpace(imgPro):
    spaceCounter = 0
    index = 0
    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter += 1
            empty[index] = True
        else:
            color = (0, 0, 255)
            thickness = 2
            empty[index] = False

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, f'{index}', (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)
                        
        index += 1

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=2, offset=20, colorR=(0,200,0))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
imgMedian = cv2.medianBlur(imgThreshold, 5)
kernel = np.ones((3, 3), np.uint8)
imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

checkParkingSpace(imgDilate)

cv2.imwrite(f"Image.png", img)
df = pd.DataFrame({
"parkVal": [i for i in range(len(posList))],
        "status": empty
})
df.to_csv('status.csv', index = False)