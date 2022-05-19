import cv2 as cv
import time
import utils

FONTS = cv.FONT_HERSHEY_COMPLEX
CLOSED_EYES_FRAME = 1
    
CALIBRATION_DURATION = 15
FACE_LOST_DETECTION = CALIBRATION_DURATION / 3

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

# landmark detection
def landmarksDetection(img, results):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]#2D landmarks
    mesh_coord_z = [(int(point.x * img_width), int(point.y * img_height), 1 / (abs(int(point.z * 100)) + 0.000001) * 100) for point in results.multi_face_landmarks[0].landmark]#3D landmarks (z coords. are not real)
    return mesh_coord, mesh_coord_z

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # RIGTH_EYE
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
    #DISTANCE
    #right eye
    rhDistance = utils.euclaideanDistance2D(rh_right, rh_left)
    rvDistance = utils.euclaideanDistance2D(rv_top, rv_bottom)
    #left eye
    lvDistance = utils.euclaideanDistance2D(lv_top, lv_bottom)
    lhDistance = utils.euclaideanDistance2D(lh_right, lh_left)
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

def Calibrate(camera, face_mesh, demonstrate):
    # arrays
    flag_calib = False
    lips_width = 1
    eyebrow_height_l = 1
    eyebrow_height_r = 1
    blinks = 1 
    average_blink_dur = 1
    
    CALIBRATING_SMILE_POSITIONS = []
    CALIBRATING_EYEBROWS_POSITIONS = []
    CALIBRATING_BLINKS_DURATION = []
    BLINKS_COUNTER = 0
    CEF_COUNTER = 0
    start_time = time.time()
    face_detection_start_timer = time.time()
    
    while True:
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        
        #  resizing frame
        frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        x = frame_width / 2
        y = frame_height / 2 
        
        if results.multi_face_landmarks:
            mesh_coords, mesh_coords_z = landmarksDetection(frame, results)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            # Calibrating happiness
            leftCorner = mesh_coords_z[LIPS[0]]
            rightCorner = mesh_coords_z[LIPS[10]]
            CALIBRATING_SMILE_POSITIONS.append(tuple([leftCorner, rightCorner]))
            
            # Calibrating amaze
            #temp_eyebrow_height_r = utils.euclaideanDistance3D(mesh_coords_z[4], mesh_coords_z[193])
            #temp_eyebrow_height_l = utils.euclaideanDistance3D(mesh_coords_z[4], mesh_coords_z[417])
            temp_eyebrow_height_l = utils.euclaideanDistance2D(mesh_coords[252], mesh_coords[295])
            temp_eyebrow_height_r = utils.euclaideanDistance2D(mesh_coords[22], mesh_coords[65])
            CALIBRATING_EYEBROWS_POSITIONS.append(tuple([temp_eyebrow_height_l, temp_eyebrow_height_r]))
            
            # Calibrating tiredness
            if CEF_COUNTER == 1: timer_started = time.time_ns() / 1000
            if ratio > 5.5:
                CEF_COUNTER +=1
            else:
                if CEF_COUNTER >= CLOSED_EYES_FRAME:
                    BLINKS_COUNTER +=1
                    CALIBRATING_BLINKS_DURATION.append(time.time_ns() / 1000 - timer_started)
                    timer_started = time.time_ns() / 1000
                    CEF_COUNTER =0
                    
            remaining_time = CALIBRATION_DURATION - (time.time()-start_time)
            if remaining_time <= 0:
                flag_calib = True
                break
            cv.circle(frame, center = (round(x), round(y)), radius =100, color =utils.RED, thickness=-1)
            utils.colorBackgroundText(frame, f"Please, look at the red circle and don't move. Remaining time: {round(remaining_time)}", FONTS, 1, (round(x - 570), round(y) - 200), 2, utils.RED, utils.YELLOW, 8, 8)
        else:
            # cannot detect face
            lost_face_detec_dur = time.time() - face_detection_start_timer
            if lost_face_detec_dur >= FACE_LOST_DETECTION:
                face_detection_start_timer = time.time()
                print ("\n\n\nError: face is not detected, trying again\n\n\n")
                CALIBRATING_SMILE_POSITIONS.clear()
                CALIBRATING_EYEBROWS_POSITIONS.clear()
                CALIBRATING_BLINKS_DURATION.clear()
                start_time = time.time()
                BLINKS_COUNTER = 0
                CEF_COUNTER = 0            
                cv.destroyAllWindows()
                continue
                
        if demonstrate is True: cv.imshow('Calibrating', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key==ord('Q'):
            break
    
    cv.destroyAllWindows()
    
    #average
    if len(CALIBRATING_SMILE_POSITIONS) > 0:
        lips_width = sum([utils.euclaideanDistance2D(i[0][:2], i[1][:2]) for i in CALIBRATING_SMILE_POSITIONS]) / len(CALIBRATING_SMILE_POSITIONS)
    else:
        flag_calib = False
    if len(CALIBRATING_EYEBROWS_POSITIONS) > 0:
        eyebrow_height_l = sum([i[0] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)
        eyebrow_height_r = sum([i[1] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)
    else:
        flag_calib = False
    if CALIBRATION_DURATION > 0:
        blinks = (BLINKS_COUNTER / CALIBRATION_DURATION) * 60.0
    else:
        flag_calib = False
    if len(CALIBRATING_BLINKS_DURATION) > 0:
        average_blink_dur = sum([i for i in CALIBRATING_BLINKS_DURATION]) / len(CALIBRATING_BLINKS_DURATION)
        if average_blink_dur <= 0: average_blink_dur = 1
    else:
        flag_calib = False
    
    #printing results
    if flag_calib is True:
        print("CALIBRATING RESULTS: ")
        print("AVERAGE MOUTH LENGTH: ", lips_width)
        print("AVERAGE LEFT EYEBROW HEIGHT: ", eyebrow_height_l)
        print("AVERAGE RIGHT EYEBROW HEIGHT: ", eyebrow_height_r)
        print("AVERAGE BLINKS PER MINUTE: ", blinks)
        print("AVERAGE BLINK DURATION: ", average_blink_dur)
    return flag_calib, lips_width, eyebrow_height_l, eyebrow_height_r, blinks, average_blink_dur
