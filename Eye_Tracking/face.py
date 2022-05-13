import cv2 as cv
from cv2 import normalize
import mediapipe as mp
import time
import utils, math
from numpy import *
import vg
# global
lips_width = 0
eyebrow_height_r = 0
eyebrow_height_l = 0
iris_angle_r = 0
iris_rad_r = 0
iris_angle_l = 0
iris_rad_l = 0

max_time = 0
# variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
BLINKS_IN_MINUTE = 0

# constants
CLOSED_EYES_FRAME = 1
FONTS = cv.FONT_HERSHEY_COMPLEX
# arrays
CALIBRATING_SMILE_POSITIONS = []
CALIBRATING_EYEBROWS_POSITIONS = []

# face bounder indices 
FACE_OVAL = [ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109 ]

# lips indices for Landmarks
LIPS = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95 ]
UPPER_LIPS = [ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ] 

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
LEFT_IRIS =[474, 475, 476, 477]

# Right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
RIGHT_IRIS =[469, 470, 471, 472]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv.VideoCapture(0)

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    mesh_coord_z = [(int(point.x * img_width), int(point.y * img_height), 1 / (abs(int(point.z * 100)) + 0.000001) * 100) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord, mesh_coord_z

# Euclaidean distance 
def euclaideanDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    try:
        reRatio = rhDistance/rvDistance
    except:
        reRatio = rhDistance
    try:
        leRatio = lhDistance/lvDistance
    except:
        leRatio = lhDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    #cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    #cv.imshow('eyes draw', eyes)
    eyes[mask==0] = 155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    try:
        gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    except:
        gaussain_blur = cropped_eye
    try:
        median_blur = cv.medianBlur(gaussain_blur, 3)
    except:
        median_blur = cropped_eye

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixed part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# attentiveness
def attenCounter():
    percent = 0.1
    return percent
    
# engagement
def engageCounter():
    percent = 0.1
    return percent
    
# tiredness
def tiredCounter():
    percent = 0.1
    return percent
    
# fatigue
def fatigCounter():
    percent = 0.1
    return percent
    
# mood
def mood():
    percent = 0.1
    return percent

def smileCounter(landmarks, minLength = 10, maxLength = 50):
    leftCorner = landmarks[LIPS[0]][:2] # 61
    rightCorner = landmarks[LIPS[10]][:2] # 291

    #upCenter = landmarks[LIPS[25]][:2] # 0
    #downCenter = landmarks[LIPS[5]][:2] # 17

    lengthOfSmile = euclaideanDistance(leftCorner, rightCorner)
    #widthOfSmile = euclaideanDistance(upCenter, downCenter)

    distanceMultiplier = 1 / (abs(landmarks[4][2]) + 0.000001)
    ratio = round(((lengthOfSmile - lips_width) / maxLength) * 10000 * distanceMultiplier ** 2, 1)
    if ratio < minLength:
        ratio = 0
    if ratio > 100:
        ratio = 100
    return ratio
    
# amazement
def amazeCounter(dr, dl, min_h = 0.01, max_h = 10):
    percent = ((dr - eyebrow_height_r) + (dl - eyebrow_height_l)) / 2
    #percent = percent % 10
    if percent > max_h: percent = 100
    elif percent < min_h: percent = 0
    else: percent = round(percent / max_h, 3) * 100
    
    return percent

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = sum(first_piece==0)
    center_part = sum(second_piece==0)
    left_part = sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = '' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color

def getXYZAverage(positions):
    avg_x = sum([i[0] for i in positions]) / len(positions)
    avg_y = sum([i[1] for i in positions]) / len(positions)
    avg_z = sum([i[2] for i in positions]) / len(positions)
    return avg_x, avg_y, avg_z

def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def lostAttention(angleOX, angleOZ):
    normalizedAngleOX = abs(90 - angleOX)
    normalizedAngleOZ = abs(90 - angleOZ)
    if normalizedAngleOX > 20:
        return True
    return False


def findTiredRatio(blinks):
    normalizedBlinks = blinks - 10
    if normalizedBlinks < 0:
        normalizedBlinks = 0
    ratio = normalizedBlinks * 5
    if ratio > 100:
        ratio = 100
    return ratio


blink_counting_start_time = time.time()

with map_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, refine_landmarks = True) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    
    #calibrating
    flag_calib = False
    while True:
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        frame_counter +=1 # frame counter
        #  resizing frame
        frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords, mesh_coords_z = landmarksDetection(frame, results, False)

            # Calibrating smile
            leftCorner = mesh_coords_z[LIPS[0]]
            rightCorner = mesh_coords_z[LIPS[10]]
            CALIBRATING_SMILE_POSITIONS.append(tuple([leftCorner, rightCorner]))
            
            # Calibrating amaze
            temp_eyebrow_height_r = math.dist(mesh_coords_z[4], mesh_coords_z[193])
            temp_eyebrow_height_l = math.dist(mesh_coords_z[4], mesh_coords_z[417])
            CALIBRATING_EYEBROWS_POSITIONS.append(tuple([temp_eyebrow_height_l, temp_eyebrow_height_r]))

        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        remaining_time = 10.0 - (time.time()-start_time)
        if remaining_time <= 0:
            flag_calib = True
            break
        #frame =utils.textWithBackground(frame, f'Please, look at the red circle and dont move. Remaining time: {round(remaining_time)}', FONTS, 1.0, (200, 200), bgOpacity=0.9, textThickness=2)
        frame =utils.textWithBackground(frame, f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        x = frame_width / 2
        y = frame_height / 2 
        cv.circle(frame, center = (round(x), round(y)), radius =100, color =utils.RED, thickness=-1)
        utils.colorBackgroundText(frame, f"Please, look at the red circle and dont move. Remaining time: {round(remaining_time)}", FONTS, 1, (round(x - 550), round(y) - 200), 2, (0,255,0), utils.YELLOW, 8, 8)
        # writing image for thumbnail drawing shape
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            flag_calib = True
            break
        
    lips_width = sum([euclaideanDistance(i[0][:2], i[1][:2]) for i in CALIBRATING_SMILE_POSITIONS]) / len(CALIBRATING_SMILE_POSITIONS)
    eyebrow_height_l = sum([i[0] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)
    eyebrow_height_r = sum([i[1] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)

    print("CALIBRATING RESULTS:")
    print("AVERAGE MOUTH LENGTH:", lips_width)
    print("AVERAGE LEFT EYEBROW HEIGHT:", eyebrow_height_l)
    print("AVERAGE RIGHT EYEBROW HEIGHT:", eyebrow_height_r)
    
    if flag_calib is True:
        while True:
            frame_counter +=1 # frame counter
            ret, frame = camera.read() # getting frame from camera 
            if not ret: 
                break # no more frames break
            
            #  resizing frame
            frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords, mesh_coords_z = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                if CEF_COUNTER == 1: timer_started = time.time_ns() / 1000
                if ratio > 5.5:
                    CEF_COUNTER +=1
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6)
    
                else:
                    if CEF_COUNTER>=CLOSED_EYES_FRAME:
                        TOTAL_BLINKS +=1
                        if max_time < (time.time_ns() / 1000 - timer_started): max_time = time.time_ns() / 1000 - timer_started
                        CEF_COUNTER =0
                utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                utils.colorBackgroundText(frame,  f'Max time mlsec: {max_time}', FONTS, 0.7, (200,100),2)
                
                cv.polylines(frame,  [array([mesh_coords[p] for p in LEFT_EYE ], dtype=int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [array([mesh_coords[p] for p in RIGHT_EYE ], dtype=int32)], True, utils.GREEN, 1, cv.LINE_AA)
                
                cv.polylines(frame,  [array([mesh_coords[p] for p in LOWER_LIPS ], dtype=int32)], True, utils.BLUE, 1, cv.LINE_AA)
                cv.polylines(frame,  [array([mesh_coords[p] for p in UPPER_LIPS ], dtype=int32)], True, utils.BLUE, 1, cv.LINE_AA)
                
                cv.polylines(frame,  [array([mesh_coords[p] for p in LEFT_EYEBROW ], dtype=int32)], True, utils.RED, 1, cv.LINE_AA)
                cv.polylines(frame,  [array([mesh_coords[p] for p in RIGHT_EYEBROW ], dtype=int32)], True, utils.RED, 1, cv.LINE_AA)
                
                cv.polylines(frame,  [array([mesh_coords[p] for p in LEFT_IRIS ], dtype=int32)], True, utils.PINK, 1, cv.LINE_AA)
                cv.polylines(frame,  [array([mesh_coords[p] for p in RIGHT_IRIS ], dtype=int32)], True, utils.PINK, 1, cv.LINE_AA)
                
                #face vertical line
                #cv.line(frame,  mesh_coords[151], mesh_coords[175], utils.GREEN, 2)
                
                #horizontal line
                p1 = (0, int(frame_height / 2))
                p2 = (int(frame_width), int(frame_height / 2))
                vec3 = array([0, 0, 1000])
                #cv.line(frame, p1, p2, utils.RED, 2)
                
                #angle between horizontal and face vertical lines
                vec1 = array([p2[0] - p1[0], p2[1] - p1[1], 0])
                vec2 = array([mesh_coords[175][0] - mesh_coords[151][0], mesh_coords[175][1] - mesh_coords[151][1], mesh_coords_z[175][2] - mesh_coords_z[151][2]])
                
                angle_h = vg.angle(vec1, vec2, look=vg.basis.z)
                angle_v = vg.angle(vec2, vec3, look=vg.basis.y)

                if (lostAttention(angleOX=angle_h, angleOZ=angle_v)):
                    utils.colorBackgroundText(frame,  f'LOST ATTENTION', FONTS, 1.7, (int(frame_height/2), 600), 2, utils.YELLOW, pad_x=6, pad_y=6)

                utils.colorBackgroundText(frame, f'Hor_Angle: {round(angle_h, 0)}', FONTS, 1.0, (frame_width - 250, 200), 2, (0,255,0), utils.RED, 8, 8)
                utils.colorBackgroundText(frame, f'Ver_Angle: {round(angle_v, 0)}', FONTS, 1.0, (frame_width - 250, 250), 2, (0,255,0), utils.RED, 8, 8)
                
                #eyebrows
                if eyebrow_height_r == -1000:
                    eyebrow_height_r = math.dist(mesh_coords_z[4], mesh_coords_z[193])
                if eyebrow_height_l == -1000:
                    eyebrow_height_l = math.dist(mesh_coords_z[4], mesh_coords_z[417])
                if lips_width == -1000:
                    lips_width = smileCounter(mesh_coords_z)
                #iris
                r_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
                l_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
                
                r = seg_intersect(array(r_iris_coords[1]),array(r_iris_coords[3]), array(r_iris_coords[0]),array(r_iris_coords[2]))
                r_iris_centre = [round(r[0]), round(r[1])]
                l = seg_intersect(array(l_iris_coords[1]),array(l_iris_coords[3]), array(l_iris_coords[0]),array(l_iris_coords[2]))
                l_iris_centre = [round(l[0]), round(l[1])]
                
                # Blink Detector Counter Completed
                #right_coords = [mesh_coords_z[p] for p in RIGHT_EYE]
                #left_coords = [mesh_coords_z[p] for p in LEFT_EYE]
                #crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                #center = mesh_coords_z[6]
                c_r = seg_intersect(array(mesh_coords[133]),array(mesh_coords[33]), array(mesh_coords[159]),array(mesh_coords[153]))
                center_r = [round(c_r[0]), round(c_r[1])]
                #center_r = [(mesh_coords[133][0] + mesh_coords[33][0])/2., (mesh_coords[159][1] + mesh_coords[153][1])/2.]
                c_l = seg_intersect(array(mesh_coords[362]),array(mesh_coords[263]), array(mesh_coords[385]),array(mesh_coords[374]))
                center_l = [round(c_l[0]), round(c_l[1])]
                #center_l = [(mesh_coords[362][0] + mesh_coords[263][0])/2., (mesh_coords[385][1] + mesh_coords[374][1])/2.]
                right_iris = [r_iris_centre[0] - center_r[0], r_iris_centre[1] - center_r[1]]
                left_iris = [l_iris_centre[0] - center_l[0], l_iris_centre[1] - center_l[1]]
                #print("l: ")
                #print(left_iris)
                #eye_position_right, color = positionEstimator(crop_right)
                #utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                utils.colorBackgroundText(frame, f'R_i: {right_iris}', FONTS, 1.0, (40, 270), 2, (0,255,0), utils.RED, 8, 8)
                #eye_position_left, color = positionEstimator(crop_left)
                #utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 370), 2, color[0], color[1], 8, 8)
                utils.colorBackgroundText(frame, f'L_i: {left_iris}', FONTS, 1.0, (40, 300), 2, (0,255,0), utils.RED, 8, 8)
                
                r_size = [euclaideanDistance(mesh_coords[133], mesh_coords[33]), euclaideanDistance(mesh_coords[159], mesh_coords[153])]
                l_size = [euclaideanDistance(mesh_coords[362], mesh_coords[263]), euclaideanDistance(mesh_coords[385], mesh_coords[374])]
                r_prop_k = [frame_width / r_size[0], frame_height / r_size[1]]
                l_prop_k = [frame_width / l_size[0], frame_height / l_size[1]]
                center_screen = [frame_width / 2., frame_height / 2.]
                screen_iris_r = [center_screen[0] + (right_iris[0] * r_prop_k[0]), center_screen[1] + (right_iris[1] * r_prop_k[1])]
                screen_iris_l = [center_screen[0] + left_iris[0] * l_prop_k[1], center_screen[1] + left_iris[1] * l_prop_k[1]]
                cv.circle(frame, center = (round(screen_iris_r[0]), round(screen_iris_r[1])), radius = 5, color = utils.RED, thickness=-1)
                cv.circle(frame, center = (round(screen_iris_l[0]), round(screen_iris_l[1])), radius = 5, color = utils.RED, thickness=-1)
                
                smile = smileCounter(mesh_coords_z)
                utils.colorBackgroundText(frame, f'Smile: {smile}', FONTS, 1.0, (frame_width - 250, 50), 2, (0,255,0), utils.RED, 8, 8)
                amaze = amazeCounter(math.dist(mesh_coords_z[4], mesh_coords_z[193]), math.dist(mesh_coords_z[4], mesh_coords_z[417]))
                utils.colorBackgroundText(frame, f'Amaze: {amaze}', FONTS, 1.0, (frame_width - 250, 100), 2, (0,255,0), utils.RED, 8, 8)

                if time.time() - blink_counting_start_time >= 60:
                    BLINKS_IN_MINUTE = TOTAL_BLINKS - BLINKS_IN_MINUTE
                    blink_counting_start_time = time.time()
                
                tiredRatio = findTiredRatio(BLINKS_IN_MINUTE)
                utils.colorBackgroundText(frame, f'Blinks in minute: {BLINKS_IN_MINUTE}', FONTS, 1.0, (50, 200), 2, (0,255,0), utils.RED, 8, 8)
                utils.colorBackgroundText(frame, f'Tired: {tiredRatio}%', FONTS, 1.0, (50, 600), 2, (0,255,0), utils.RED, 8, 8)
    
            # calculating  frame per seconds FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time
    
            frame = utils.textWithBackground(frame,f'FPS: {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            # writing image for thumbnail drawing shape
            #cv.imwrite(f'img/frame_{frame_counter}.png', frame)
            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key==ord('q') or key ==ord('Q'):
                break
    cv.destroyAllWindows()
    camera.release()
