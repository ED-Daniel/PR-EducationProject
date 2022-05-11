from ctypes import util
from turtle import pen, width, window_width
import cv2 as cv
import mediapipe as mp
import time
from main import euclaideanDistance
import utils, math
import numpy as np
# variables 
frame_counter =0
# constants
FONTS =cv.FONT_HERSHEY_COMPLEX

HOLD_SMILE_FRAMES = 0
CALIBRATING_SMILE_FRAMES = 300
CALIBRATING_SMILE_POSITIONS = []

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37, 0,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(0)
# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    mesh_coord_z = [(int(point.x * img_width), int(point.y * img_height), int(point.z * 1000)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord, mesh_coord_z

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords, mesh_coords_z = landmarksDetection(frame, results, False)

            # Calibrating smile
            leftCorner = mesh_coords_z[LIPS[0]]
            rightCorner = mesh_coords_z[LIPS[10]]

            HOLD_SMILE_FRAMES += 1
            CALIBRATING_SMILE_POSITIONS.append(tuple([leftCorner, rightCorner]))

            if HOLD_SMILE_FRAMES >= CALIBRATING_SMILE_FRAMES:
                break

        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        utils.textBlurBackground(frame, "Calibrating", FONTS, 1.0, (350, 50))
        # writing image for thumbnail drawing shape
        cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()


def getXYZAverage(positions):
    avg_x = sum([i[0] for i in positions]) / len(positions)
    avg_y = sum([i[1] for i in positions]) / len(positions)
    avg_z = sum([i[2] for i in positions]) / len(positions)
    return avg_x, avg_y, avg_z

avg_left = getXYZAverage([i[0] for i in CALIBRATING_SMILE_POSITIONS])
avg_right = getXYZAverage([i[1] for i in CALIBRATING_SMILE_POSITIONS])
avg_length = sum([euclaideanDistance(i[0][:2], i[1][:2]) for i in CALIBRATING_SMILE_POSITIONS]) / len(CALIBRATING_SMILE_POSITIONS)

print("CALIBRATING RESULTS:")
print("AVERAGE LEFT MOUTH POSITION:")
print("X:", avg_left[0], "Y:", avg_left[1], "Z:", avg_left[2], sep="\t")
print("AVERAGE RIGHT MOUTH POSITION:")
print("X:", avg_right[0], "Y:", avg_right[1], "Z:", avg_right[2], sep="\t")
print("AVERAGE LENGTH:", avg_length)