import install_requirements

import cv2
from faceDetectionModule import faceDetection

video = 'C:/Users/buysa/OneDrive/Documents/pythonProjects/blink_counter/Recording.mp4'
cap = cv2.VideoCapture(video)
cap.set(3,640)
cap.set(4, 480)

detector = faceDetection(minDetectionCon=0.75)

run = True
while run:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=True)

    if bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox['bbox']

            # To avoid the error when leaving the area
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            face = img[y:y + h, x:x + w]
            bluredFace = cv2.blur(face, (77,77))
            img[y:y + h, x:x + w] = bluredFace

    img = cv2.resize(img, (640, 360))
    cv2.imshow('Image', img)
    cv2.waitKey(1)