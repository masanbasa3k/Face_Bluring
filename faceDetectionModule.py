import cv2
import mediapipe as mp

class faceDetection:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        for id, detection in enumerate(self.results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            bboxInfo = {"id": id, "bbox": bbox, "score": detection.score[0], "center": (cx, cy)}
            bboxs.append(bboxInfo)
            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs