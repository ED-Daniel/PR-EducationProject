from tkinter import LEFT
import cv2 as cv
import mediapipe as mp
import time
from face import landmarksDetection
import utils, math
import numpy as np
from constants import RIGHT_EYE, LEFT_EYE, LIPS

class Detector:
    def __init__(self) -> None:
        self.map_face_mesh = mp.solutions.face_mesh
        self.camera = cv.VideoCapture(0)
        self.image = self.camera.read()[1]
        self.coords = []

        self.frame_counter =0
        self.CEF_COUNTER =0
        self.TOTAL_BLINKS =0
        self.CLOSED_EYES_FRAME = 1
        self.FONTS = cv.FONT_HERSHEY_COMPLEX

    def landmarksDetection(self, results, draw=False):
        img_height, img_width = self.image.shape[:2]
        mesh_coord_z = [(int(point.x * img_width), int(point.y * img_height), int(point.z * 1000)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            [cv.circle(self.image, p[:2], 2, (0,255,0), -1) for p in mesh_coord_z]
        self.coords = mesh_coord_z
    
    def euclaideanDistance(self, point, point1):
        x, y = point[:2]
        x1, y1 = point1[:2]
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance
    
    def blinkRatio(self):
        rh_right = self.coords[RIGHT_EYE[0]]
        rh_left = self.coords[RIGHT_EYE[8]]
 
        rv_top = self.coords[RIGHT_EYE[12]]
        rv_bottom = self.coords[RIGHT_EYE[4]]

        lh_right = self.coords[LEFT_EYE[0]]
        lh_left = self.coords[LEFT_EYE[8]]

        lv_top = self.coords[LEFT_EYE[12]]
        lv_bottom = self.coords[LEFT_EYE[4]]

        rhDistance = self.euclaideanDistance(rh_right, rh_left)
        rvDistance = self.euclaideanDistance(rv_top, rv_bottom)

        lvDistance = self.euclaideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclaideanDistance(lh_right, lh_left)

        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance

        ratio = (reRatio + leRatio) / 2
        return ratio 

    def getSmile(self, minLength=50, maxLength=130):
        leftCorner = self.coords[LIPS[0]][:2] # 61
        rightCorner = self.coords[LIPS[10]][:2] # 291

        upCenter = self.coords[LIPS[25]][:2] # 0
        downCenter = self.coords[LIPS[5]][:2] # 17

        lengthOfSmile = self.euclaideanDistance(leftCorner, rightCorner)
        widthOfSmile = self.euclaideanDistance(upCenter, downCenter)

        distanceMultiplier = 1 / (abs(self.coords[4][2]) + 0.000001)
        ratio = round(((lengthOfSmile - minLength) / maxLength) * 10000 * distanceMultiplier ** 2, 1)
        if ratio < 0:
            ratio = 0
        if ratio > 100:
            ratio = 100
        return ratio
    
    def eyesExtractor(self):
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        dim = gray.shape
        mask = np.zeros(dim, dtype=np.uint8)

        rightEyeCoords = [self.coords[i][:2] for i in RIGHT_EYE]
        leftEyeCoords = [self.coords[i][:2] for i in LEFT_EYE]
 
        cv.fillPoly(mask, [np.array(rightEyeCoords, dtype=np.int32)], 255)
        cv.fillPoly(mask, [np.array(leftEyeCoords, dtype=np.int32)], 255)

        # cv.imshow('mask', mask)

        eyes = cv.bitwise_and(gray, gray, mask=mask)
        # cv.imshow('eyes draw', eyes)
        eyes[mask==0]=155

        # getting minium and maximum x and y  for right and left eyes 
        # For Right Eye 
        r_max_x = (max(rightEyeCoords, key=lambda item: item[0]))[0]
        r_min_x = (min(rightEyeCoords, key=lambda item: item[0]))[0]
        r_max_y = (max(rightEyeCoords, key=lambda item: item[1]))[1]
        r_min_y = (min(rightEyeCoords, key=lambda item: item[1]))[1]

        # For LEFT Eye
        l_max_x = (max(leftEyeCoords, key=lambda item: item[0]))[0]
        l_min_x = (min(leftEyeCoords, key=lambda item: item[0]))[0]
        l_max_y = (max(leftEyeCoords, key=lambda item: item[1]))[1]
        l_min_y = (min(leftEyeCoords, key=lambda item: item[1]))[1]

        # croping the eyes from mask 
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

        # returning the cropped eyes 
        return cropped_right, cropped_left

    def positionEstimator(self, cropped_eye):
        h, w =cropped_eye.shape

        try:
            gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
        except:
            gaussain_blur = cropped_eye
        try:
            median_blur = cv.medianBlur(gaussain_blur, 3)
        except:
            median_blur = cropped_eye

        ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

        piece = int(w/3) 

        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]

        eye_position, color = self.pixelCounter(right_piece, center_piece, left_piece)

        return eye_position, color

    def pixelCounter(self, first_piece, second_piece, third_piece):
        right_part = np.sum(first_piece==0)
        center_part = np.sum(second_piece==0)
        left_part = np.sum(third_piece==0)

        eye_parts = [right_part, center_part, left_part]

        max_index = eye_parts.index(max(eye_parts))
        pos_eye = '' 
        if max_index == 0:
            pos_eye = "RIGHT"
            color = [utils.BLACK, utils.GREEN]
        elif max_index == 1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]
        elif max_index == 2:
            pos_eye = 'LEFT'
            color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye = "Closed"
            color = [utils.GRAY, utils.YELLOW]

        return pos_eye, color

    def run(self):
        with self.map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            start_time = time.time()

            while True:
                self.frame_counter +=1
                ret, frame = self.camera.read() 
                self.image = frame
                if not ret: 
                    break

                frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                frame_height, frame_width= frame.shape[:2]
                rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                results  = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    self.landmarksDetection(results, False)
                    ratio = self.blinkRatio()
                    utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', self.FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

                    if ratio >5.5:
                        self.CEF_COUNTER +=1
                        utils.colorBackgroundText(frame,  f'Blink', self.FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    else:
                        if self.CEF_COUNTER > self.CLOSED_EYES_FRAME:
                            self.TOTAL_BLINKS +=1
                            self.CEF_COUNTER =0
                    utils.colorBackgroundText(frame,  f'Total Blinks: {self.TOTAL_BLINKS}', self.FONTS, 0.7, (30,150),2)

                    cv.polylines(frame, [np.array([self.coords[p][:2] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame, [np.array([self.coords[p][:2] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                    # Blink Detector Counter Completed
                    crop_right, crop_left = self.eyesExtractor()

                    eye_position, color = self.positionEstimator(crop_right)
                    utils.colorBackgroundText(frame, f'R: {eye_position}', self.FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                    eye_position_left, color =  self.positionEstimator(crop_left)
                    utils.colorBackgroundText(frame, f'L: {eye_position_left}', self.FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)

                    smileWidth = self.getSmile()
                    utils.colorBackgroundText(frame, f'Smile width: {smileWidth}', self.FONTS, 1.0, (40, 420), 2, color[0], color[1], 8, 8)
                    utils.colorBackgroundText(frame, f'From camera: {self.coords[0][2]}', self.FONTS, 1.0, (40, 520), 2, color[0], color[1], 8, 8)
                
                end_time = time.time() - start_time
                fps = self.frame_counter / end_time

                frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}', self.FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
                # writing image for thumbnail drawing shape
                cv.imwrite(f'img/frame_{self.frame_counter}.png', frame)
                cv.imshow('frame', frame)
                key = cv.waitKey(2)
                if key==ord('q') or key ==ord('Q'):
                    break

            cv.destroyAllWindows()
            self.camera.release()
