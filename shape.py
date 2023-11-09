import numpy as np 
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 620, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 67, empty)         # 67
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)        # 255
cv2.createTrackbar("Area", "Parameters", 30000, 27000, empty)          #  


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, original, imgContour):
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        minArea = cv2.getTrackbarPos("Area", "Parameters")
        if area > minArea:
            cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))  
            vertices = np.array([approx[0], approx[1], approx[2], approx[3]], dtype=np.int32)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, [vertices], (255, 255, 255))  
            result = cv2.bitwise_and(img, mask)              
            # if len(approx) >= 4:
            #     cv2.imwrite("ROI.jpeg", original[approx[0], approx[1], approx[2], approx[3]])


            cv2.imwrite("ROI.jpeg", result)

            # print vertices
            # for a in approx:
            #     print(a)

            x, y, w, h = cv2.boundingRect(approx)
            roi = original[y:y+h, x:x+w]
            cv2.imwrite("ROI_2.jpeg", roi)

while True:
    success, img = cap.read()

    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2, 7) # 7, 5 or 3 

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, img, imgContour)

    # cv2.imshow("ROI", img[approx[0], approx[1], approx[2], approx[3]])

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny],
                                 [imgDil, imgContour, imgContour]))
    cv2.imshow("Webcam", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
