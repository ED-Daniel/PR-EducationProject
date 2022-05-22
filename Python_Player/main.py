import cv2 as cv
import mediapipe as mp
import time
import utils, calibrating
import numpy as np
import vg
import sys

# global fields
lips_width = 1
eyebrow_height_r = 1
eyebrow_height_l = 1
blinks = 1
average_blink_dur = 1
average_between_blinks_dur = 1

# variables 
HEAD_CHECK_TIMER = 30
HEAD_ANGLE = 30
RECALIB_TIME = 180
EYES_CLOSE_DURATION = 5
RECALIBRATION_DURATION = 15
FACE_LOST_DETECTION = 10
MAX_BLINK_DURATION = 5
TIREDNESS_DURATION = 60
ENGAGEMENT_DURATION = 60
RATIO = 5.5

# constants
CLOSED_EYES_FRAME = 1
FONTS = cv.FONT_HERSHEY_COMPLEX

# const
KWIDTH = 1.2
KHEIGHT = 2
KTRACK = 0.7
KDETECT = 0.7
KVISUAL_RAD = 0.8
KAMAZE = 0.65
KHAPPY = 0.2

# face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LEFT_IRIS = [474, 475, 476, 477]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_IRIS = [469, 470, 471, 472]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv.VideoCapture(0)


# landmark detection
def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]  # 2D landmarks
    mesh_coord_z = [
        (int(point.x * img_width), int(point.y * img_height), 1 / (abs(int(point.z * 100)) + 0.000001) * 100) for point
        in results.multi_face_landmarks[0].landmark]  # 3D landmarks (z coords. are not real)
    return mesh_coord, mesh_coord_z

  
# blinking Ratio
def blinkRatio(img, landmarks):
    # RIGTH_EYE
    # horizontal line 
    rh_right = landmarks[RIGHT_EYE[0]]
    rh_left = landmarks[RIGHT_EYE[8]]
    # vertical line 
    rv_top = landmarks[RIGHT_EYE[12]]
    rv_bottom = landmarks[RIGHT_EYE[4]]
    
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[LEFT_EYE[0]]
    lh_left = landmarks[LEFT_EYE[8]]
    # vertical line 
    lv_top = landmarks[LEFT_EYE[12]]
    lv_bottom = landmarks[LEFT_EYE[4]]
    
    #DISTANCE
    #right eye
    rhDistance = utils.euclaideanDistance2D(rh_right, rh_left)
    rvDistance = utils.euclaideanDistance2D(rv_top, rv_bottom)
    # left eye
    lvDistance = utils.euclaideanDistance2D(lv_top, lv_bottom)
    lhDistance = utils.euclaideanDistance2D(lh_right, lh_left)
    try:
        reRatio = rhDistance / rvDistance
    except:
        reRatio = rhDistance
    try:
        leRatio = lhDistance / lvDistance
    except:
        leRatio = lhDistance
    ratio = (reRatio + leRatio) / 2
    return ratio


# attention tracking
def lostAttention(point_ir, point_center, rad, eyes_closed):
    #normalizedAngleOX = abs(90 - angleOX)
    flag = False
    if (point_ir[0] - point_center[0])**2 + (point_ir[1] - point_center[1])**2 <= rad**2: flag = True
    if flag is False or eyes_closed > EYES_CLOSE_DURATION is True:
        return True
    return False

# tiredness counter
def tiredCounter(bl, dur):
    norm_bl = bl - blinks
    if norm_bl < 0: norm_bl = 0
    norm_dur = dur - average_blink_dur
    if norm_dur < 0: norm_dur = 0
    ratio = (norm_bl * 80 / blinks) + (norm_dur * 20 / average_blink_dur)
    if ratio > 100:
        ratio = 100
    if ratio < 0:
        ratio = 0
    return ratio

# engagement counter
def engageCounter(bl, dur, fl):
    percent = 0.7 * (100 - (bl / blinks * 100)) + 0.3 * (dur / average_between_blinks_dur * 100 - 100) - (fl * 30)
    if percent > 100:
        percent = 100
    elif percent < 0:
        percent = 0
    return percent

# happiness counter
def happyCounter(landmarks, dur):
    minLength = lips_width
    maxLength = lips_width * KHAPPY
    leftCorner = landmarks[LIPS[0]][:2]
    rightCorner = landmarks[LIPS[10]][:2]
    #upCenter = landmarks[LIPS[25]][:2]
    #downCenter = landmarks[LIPS[5]][:2]
    lengthOfSmile = utils.euclaideanDistance2D(leftCorner, rightCorner)
    percent = 0
    if dur >= 4:
        percent = round((lengthOfSmile - minLength) / maxLength * 100)
    if percent < 0: percent = 0
    elif percent > 100: percent = 100
    return percent
    
# amazement counter
def amazeCounter(dl, dr, angle):
    minLength = (eyebrow_height_l + eyebrow_height_r) / 2
    # max_h = minLength * 0.045
    max_h = minLength * KAMAZE
    percent = round((((dr + dl) / 2) - minLength) / max_h * 100)

    if angle > 30: percent -= 20
    if percent > 100: percent = 100
    elif percent < 20: percent = 0
    return percent

with map_face_mesh.FaceMesh(min_detection_confidence = KDETECT, min_tracking_confidence = KTRACK, refine_landmarks = True) as face_mesh:
    # calibration
    flag_calib = False
    flag_calib, lips_width, eyebrow_height_l, eyebrow_height_r, blinks, average_blink_dur, average_between_blinks_dur = calibrating.Calibrate(camera, face_mesh, True)
    
    # starting timers
    blink_counting_start_time = time.time()
    fps_start_time = time.time()
    head_fluctuation_counting_start_timer = time.time()
    recalibration_start_timer = time.time()
    recalibrating_start_timer = 0
    blinks_start_timer = 0
    recalibrating_blinks_start_timer = 0
    recalibrating_between_blinks_start_timer = 0
    face_detection_start_timer = time.time()
    between_blink_start_timer = time.time_ns() / 1000
    dur_between_blink_start_timer = time.time()
    head_check_start_time = time.time()
    
    if flag_calib is True:
        
        LAST_DURATION = 0
        AV_BLINK_DURATION = []
        AV_BETWEEN_BLINKS_DURATION = []
        FRAME_COUNTER = 0
        CEF_COUNTER = 0
        TOTAL_BLINKS = 0
        BLINKS_IN_TIME = 0
        HEAD_FLUCTUATION_IN_TIME = 0
        HEAD_FLUCTUATION_COUNTER = 0
        AV_DUR = average_blink_dur
        IS_RECALIBRATING = False
        HEAD_FLUCTUATION_IN_TIME = 0
        tiredRatio = 0
        engagementRatio = 0
        
        # recalibrating fields
        RECALIBRATING_SMILE_POSITIONS = []
        RECALIBRATING_EYEBROWS_POSITIONS = []
        RECALIBRATING_BLINKS_DURATION = []
        RECALIBRATING_BETWEEN_BLINKS_DURATION = []
        RECALIBRATING_BLINKS_COUNTER = 0
        RECALIBRATING_CEF_COUNTER = 0

        while True:
            FRAME_COUNTER +=1 # frame counter
            ret, frame = camera.read() # getting frame from camera 
            if not ret: 
                print("\n\n\nError with camera!\n\n\n")
                continue
            
            # resizing frame
            frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_coords, mesh_coords_z = landmarksDetection(frame, results)
                ratio = blinkRatio(frame, mesh_coords)
                utils.colorBackgroundText(frame, f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                if CEF_COUNTER == 1: 
                    blinks_start_timer = time.time_ns() / 1000
                    AV_BETWEEN_BLINKS_DURATION.append(time.time_ns() / 1000 - between_blink_start_timer)
                if ratio > RATIO:
                    CEF_COUNTER +=1
                    utils.colorBackgroundText(frame, 'BlINK', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, 6, 6)
                elif CEF_COUNTER >= CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    between_blink_start_timer = time.time_ns() / 1000
                    LAST_DURATION = time.time_ns() / 1000 - blinks_start_timer
                    CEF_COUNTER =0
                    # fail: blink is not counted
                    if LAST_DURATION < MAX_BLINK_DURATION * 1000000 : 
                        AV_BLINK_DURATION.append(LAST_DURATION)
                        BLINKS_IN_TIME +=1
                #utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                #utils.colorBackgroundText(frame,  f'Last blink duration mcsec: {round(LAST_DURATION, 2)}', FONTS, 0.7, (30,100), 2)
                
                #drawing eyes, lips, irises
# =============================================================================
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in LEFT_EYE ], utils.GREEN)
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in RIGHT_EYE ], utils.GREEN)
#                 
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in LOWER_LIPS ], utils.BLUE)
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in UPPER_LIPS ], utils.BLUE)
#                 
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in LEFT_EYEBROW ], utils.RED)
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in RIGHT_EYEBROW ], utils.RED)
#                 
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in LEFT_IRIS ], utils.PINK)
#                 utils.pollyLines(frame,  [mesh_coords[p] for p in RIGHT_IRIS ], utils.PINK)
# =============================================================================

                # head angles
                p1 = (0, int(frame_height / 2))
                p2 = (int(frame_width), int(frame_height / 2))
                vec3 = np.array([0, 0, 1000])
                
                # angle between horizontal and face vertical lines
                vec1 = np.array([p2[0] - p1[0], p2[1] - p1[1], 0])
                vec2 = np.array([mesh_coords[175][0] - mesh_coords[151][0], mesh_coords[175][1] - mesh_coords[151][1], mesh_coords_z[175][2] - mesh_coords_z[151][2]])
                angle_h = vg.angle(vec1, vec2, look =vg.basis.z)
                angle_v = vg.angle(vec2, vec3, look =vg.basis.y)
                
                # showing head angles
# =============================================================================
#                 utils.colorBackgroundText(frame, f'Hor_Angle: {round(angle_h, 0)}', FONTS, 1.0, (frame_width - 250, 250), 2, (0,255,0), utils.RED, 8, 8)
#                 utils.colorBackgroundText(frame, f'Ver_Angle: {round(angle_v, 0)}', FONTS, 1.0, (frame_width - 250, 250), 2, (0,255,0), utils.RED, 8, 8)
# =============================================================================
                
                # iris
                r_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
                l_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
                
                center_screen = [round(frame_width / 2.), round(frame_height / 2.)]
                # cv.circle(frame, center = (round(center_screen[0]), round(center_screen[1])), radius = round(frame_height / 2 * KVISUAL_RAD), color = utils.RED, thickness=2)
                # tracking iris (if not blinking)           
                if ratio <= 5.5:
                    r = utils.seg_intersect(np.array(r_iris_coords[0]), np.array(r_iris_coords[2]), np.array(r_iris_coords[1]), np.array(r_iris_coords[3]))

                    r_iris_center = [round(r[0]), round(r[1])]
                    l = utils.seg_intersect(np.array(l_iris_coords[0]), np.array(l_iris_coords[2]),
                                            np.array(l_iris_coords[1]), np.array(l_iris_coords[3]))
                    l_iris_center = [round(l[0]), round(l[1])]
                    try:
                        c_r = utils.seg_intersect(np.array(mesh_coords[173]), np.array(mesh_coords[33]), np.array(mesh_coords[159]), np.array(mesh_coords[153]))
                        center_r = [round(c_r[0]), round(c_r[1])]
                        c_l = utils.seg_intersect(np.array(mesh_coords[398]), np.array(mesh_coords[263]), np.array(mesh_coords[380]), np.array(mesh_coords[386]))
                        center_l = [round(c_l[0]), round(c_l[1])]
                        right_iris = [r_iris_center[0] - center_r[0], r_iris_center[1] - center_r[1]]
                        left_iris = [l_iris_center[0] - center_l[0], l_iris_center[1] - center_l[1]]
                        r_size = [utils.euclaideanDistance3D(mesh_coords_z[173], mesh_coords_z[33]), utils.euclaideanDistance3D(mesh_coords_z[159], mesh_coords_z[153])]
# =============================================================================
#                         cv.circle(frame, center = (round(center_r[0]), round(center_r[1])), radius = 10, color = utils.RED, thickness=2)
#                         cv.circle(frame, center = (round(center_l[0]), round(center_l[1])), radius = 10, color = utils.RED, thickness=2)
# =============================================================================
                        l_size = [utils.euclaideanDistance3D(mesh_coords_z[398], mesh_coords_z[263]), utils.euclaideanDistance3D(mesh_coords_z[380], mesh_coords_z[386])]
                        r_prop_k = [round(frame_width / r_size[0]), round(frame_height / r_size[1])]
                        l_prop_k = [round(frame_width / l_size[0]), round(frame_height / l_size[1])]
                        
                        # glance position on screen
                        screen_iris_l = [center_screen[0] - (left_iris[0] * l_prop_k[0] * KWIDTH), center_screen[1] + (left_iris[1] * l_prop_k[1] * KHEIGHT - 50)]
                        screen_iris_r = [center_screen[0] - (right_iris[0] * r_prop_k[0] * KWIDTH), center_screen[1] + (right_iris[1] * r_prop_k[1] * KHEIGHT - 50)]
                        screen_irises = [round((screen_iris_l[0] + screen_iris_r[0]) / 2), round((screen_iris_l[1] + screen_iris_r[1]) / 2)]
                        # if utils.euclaideanDistance2D(screen_iris_l, screen_iris_r) < frame_width/2: 
                        #     cv.circle(frame, center = (round(screen_irises[0]), round(screen_irises[1])), radius = 40, color = utils.RED, thickness=2)
                            
                        # iris position
      # =============================================================================
      #                 utils.colorBackgroundText(frame, f'R_i pos: {right_iris}', FONTS, 1.0, (40, 270), 2, (0,255,0), utils.RED, 8, 8)
      #                 utils.colorBackgroundText(frame, f'L_i pos: {left_iris}', FONTS, 1.0, (40, 300), 2, (0,255,0), utils.RED, 8, 8)
      # =============================================================================
                    except:
                        print("\n\n\nIris tracking error\n\n\n")    
                        
                # attention
                temp = 0
                if ratio > RATIO:
                    temp = (time.time_ns() / 1000 - blinks_start_timer) / 1000000
                if lostAttention(screen_irises, center_screen, (frame_height / 2 * KVISUAL_RAD), temp) is True:
                    # utils.colorBackgroundText(frame, 'ATTENTION LOST', FONTS, 1.7, (int(frame_width/2) - 100, int(frame_height/2)), 2, utils.YELLOW, pad_x=6, pad_y=6)
                    print('\n\n\nATTENTION LOST\n\n\n')
                    
                # happiness
                t = (time.time_ns() / 1000) - between_blink_start_timer
                happiness = happyCounter(mesh_coords_z, t / 1000000)
                # utils.colorBackgroundText(frame, f'Happiness: {happiness}%', FONTS, 1.0, (frame_width - 350, 50), 2, (0,255,0), utils.RED, 8, 8)
                print(f'Happiness: {happiness}%')
                
                # amaze
                amazement = amazeCounter(utils.euclaideanDistance2D(mesh_coords[22], mesh_coords[65]), utils.euclaideanDistance2D(mesh_coords[252], mesh_coords[295]), abs(90 - angle_h))
                # utils.colorBackgroundText(frame, f'Amazement: {amazement}%', FONTS, 1.0, (frame_width - 350, 100), 2, (0,255,0), utils.RED, 8, 8)
                print (f'Amazement: {amazement}%')
                
                # tiredness
                if time.time() - blink_counting_start_time >= TIREDNESS_DURATION:
                    if len(AV_BLINK_DURATION) == 0: AV_DUR = average_blink_dur
                    else: AV_DUR = sum([i for i in AV_BLINK_DURATION]) / len(AV_BLINK_DURATION)
                    blink_counting_start_time = time.time()
                    AV_BLINK_DURATION.clear()
                    tiredRatio = tiredCounter(BLINKS_IN_TIME, AV_DUR)
                    BLINKS_IN_TIME = 0
                # utils.colorBackgroundText(frame, f'Tiredness: {tiredRatio}%', FONTS, 1.0, (frame_width - 350, 200), 2, (0,255,0), utils.RED, 8, 8)
                print(f'Tiredness: {tiredRatio}%')
                
                # head fulctuation
                if abs(90 - angle_h) > HEAD_ANGLE and time.time() - head_check_start_time >= 1:
                    HEAD_FLUCTUATION_COUNTER += 1
                    head_check_start_time = time.time()
                if time.time() - head_fluctuation_counting_start_timer >= HEAD_CHECK_TIMER:
                    HEAD_FLUCTUATION_IN_TIME = HEAD_FLUCTUATION_COUNTER / HEAD_CHECK_TIMER
                    head_fluctuation_counting_starting_time = time.time()
                    HEAD_FLUCTUATION_COUNTER = 0
                
                # engagement
                if time.time() - dur_between_blink_start_timer >= ENGAGEMENT_DURATION and len(AV_BETWEEN_BLINKS_DURATION) > 0:
                    dur_between_blink_start_timer = time.time()
                    engagementRatio = engageCounter(BLINKS_IN_TIME * (60 / TIREDNESS_DURATION), sum([i for i in AV_BETWEEN_BLINKS_DURATION]) / len(AV_BETWEEN_BLINKS_DURATION), HEAD_FLUCTUATION_IN_TIME)
                # utils.colorBackgroundText(frame, f'Engagement: {engagementRatio}%', FONTS, 1.0, (frame_width - 350, 150), 2, (0,255,0), utils.RED, 8, 8)
                print(f'Engagement: {engagementRatio}%')
                    
                    # showing blinks in minute
# =============================================================================
#                     utils.colorBackgroundText(frame, f'Blinks in minute: {BLINKS_IN_MINUTE}', FONTS, 1.0, (30,200), 2, (0,255,0), utils.RED, 8, 8)
# =============================================================================
                    

                # recalibrating
                if RECALIBRATION_DURATION - recalibrating_start_timer <= 0 and IS_RECALIBRATING is False: 
                    temp = False

                    #average
                    if len(RECALIBRATING_SMILE_POSITIONS) > 0:
                        lips_width = sum([utils.euclaideanDistance2D(i[0][:2], i[1][:2]) for i in RECALIBRATING_SMILE_POSITIONS]) / len(RECALIBRATING_SMILE_POSITIONS)
                        temp = True
                    else:
                        temp = False
                    if len(RECALIBRATING_EYEBROWS_POSITIONS) > 0:
                        eyebrow_height_l = sum([i[0] for i in RECALIBRATING_EYEBROWS_POSITIONS]) / len(RECALIBRATING_EYEBROWS_POSITIONS)
                        eyebrow_height_r = sum([i[1] for i in RECALIBRATING_EYEBROWS_POSITIONS]) / len(RECALIBRATING_EYEBROWS_POSITIONS)
                        temp = True
                    else:
                        temp = False
                    if RECALIBRATION_DURATION > 0:
                        blinks = (RECALIBRATING_BLINKS_COUNTER / RECALIBRATION_DURATION) * 60.0
                        temp = True
                    else: 
                        temp = False
                    if len(RECALIBRATING_BLINKS_DURATION) > 0:
                        average_blink_dur = sum([i for i in RECALIBRATING_BLINKS_DURATION]) / len(RECALIBRATING_BLINKS_DURATION)
                        if average_blink_dur <= 0: average_blink_dur = 1
                        temp = True
                    else:
                        temp = False
                    if len(RECALIBRATING_BETWEEN_BLINKS_DURATION) > 0:
                        average_between_blinks_dur = sum([i for i in RECALIBRATING_BETWEEN_BLINKS_DURATION]) / len(RECALIBRATING_BETWEEN_BLINKS_DURATION)
                        if average_between_blinks_dur <= 0: average_between_blinks_dur = 1
                        temp = True
                    else:
                        temp = False
                        
                    if temp is True:
                        #printing recalibration results
                        print("RECALIBRATING RESULTS: ")
                        print("NEW AVERAGE MOUTH LENGTH: ", lips_width)
                        print("NEW AVERAGE LEFT EYEBROW HEIGHT: ", eyebrow_height_l)
                        print("NEW AVERAGE RIGHT EYEBROW HEIGHT: ", eyebrow_height_r)
                        print("NEW AVERAGE BLINKS PER MINUTE: ", blinks)
                        print("NEW AVERAGE BLINK DURATION: ", average_blink_dur)
                        print("NEW AVERAGE BETWEEN BLINK DURATION: ", average_between_blinks_dur)
                        recalibration_start_timer = time.time()
                        recalibrating_start_timer = 0
                    else:
                        print("\n\n\nError: recalibrating has failed, restarting...\n\n\n")
                        IS_RECALIBRATING = True
                        recalibrating_start_timer = time.time()
                        
                    RECALIBRATING_SMILE_POSITIONS.clear()
                    RECALIBRATING_EYEBROWS_POSITIONS.clear()
                    RECALIBRATING_BLINKS_DURATION.clear()
                    RECALIBRATING_BLINKS_COUNTER = 0
                    RECALIBRATING_CEF_COUNTER = 0 
                    recalibrating_blinks_start_timer = time.time_ns() / 1000
                    recalibrating_between_blinks_start_timer = time.time_ns() / 1000000
                      
                elif time.time() - recalibration_start_timer >= RECALIB_TIME and IS_RECALIBRATING is False: 
                    IS_RECALIBRATING = True
                    RECALIBRATING_SMILE_POSITIONS.clear()
                    RECALIBRATING_EYEBROWS_POSITIONS.clear()
                    RECALIBRATING_BLINKS_DURATION.clear()
                    RECALIBRATING_BLINKS_COUNTER = 0
                    RECALIBRATING_CEF_COUNTER = 0 
                    recalibrating_blinks_start_timer = time.time_ns() / 1000
                    recalibrating_between_blinks_start_timer = time.time_ns() / 1000000
                    recalibrating_start_timer = time.time()

                elif IS_RECALIBRATING is True:
                    # Calibrating happiness
                    leftCorner = mesh_coords_z[LIPS[0]]
                    rightCorner = mesh_coords_z[LIPS[10]]
                    RECALIBRATING_SMILE_POSITIONS.append(tuple([leftCorner, rightCorner]))
                                        
                    # Calibrating amaze
                    temp_eyebrow_height_l = utils.euclaideanDistance2D(mesh_coords[252], mesh_coords[295])
                    temp_eyebrow_height_r = utils.euclaideanDistance2D(mesh_coords[22], mesh_coords[65])
                    # temp_eyebrow_height_l = utils.euclaideanDistance3D(mesh_coords_z[252], mesh_coords_z[295])
                    # temp_eyebrow_height_r = utils.euclaideanDistance3D(mesh_coords_z[22], mesh_coords_z[65])
                    RECALIBRATING_EYEBROWS_POSITIONS.append(tuple([temp_eyebrow_height_l, temp_eyebrow_height_r]))
                                   
                    recalibrating_ratio = blinkRatio(frame, mesh_coords)
                    
                    # Calibrating tiredness
                    if RECALIBRATING_CEF_COUNTER == 1: recalibrating_blinks_start_timer = time.time_ns() / 1000
                    if recalibrating_ratio > RATIO:
                        RECALIBRATING_CEF_COUNTER +=1
                        RECALIBRATING_BETWEEN_BLINKS_DURATION.append(time.time_ns() / 1000 - recalibrating_between_blinks_start_timer)
                        recalibrating_between_blinks_start_timer = time.time_ns() / 1000000
                    else:
                        if RECALIBRATING_CEF_COUNTER >= CLOSED_EYES_FRAME:
                            RECALIBRATING_BLINKS_COUNTER +=1
                            RECALIBRATING_BLINKS_DURATION.append(time.time_ns() / 1000 - recalibrating_blinks_start_timer)
                            recalibrating_blinks_start_timer = time.time_ns() / 1000
                            RECALIBRATING_BLINKS_COUNTER =0
                    recalibrating_duration_remaining = RECALIBRATION_DURATION - (time.time() - recalibrating_start_timer)
                    if recalibrating_duration_remaining <= 0:
                        IS_RECALIBRATING = False
                    
                #utils.colorBackgroundText(frame, f'Head fluctuation per 30s: {HEAD_FLUCTUATION_IN_TIME}%', FONTS, 1.0, (10, 250), 2, (0,255,0), utils.RED, 8, 8)
                
            else:
                # cannot detect face
                lost_face_detec_dur = time.time() - face_detection_start_timer
                
                if lost_face_detec_dur >= FACE_LOST_DETECTION:
                    IS_RECALIBRATING = False
                    RECALIBRATING_SMILE_POSITIONS.clear()
                    RECALIBRATING_EYEBROWS_POSITIONS.clear()
                    RECALIBRATING_BLINKS_DURATION.clear()
                    RECALIBRATING_BLINKS_COUNTER = 0
                    RECALIBRATING_CEF_COUNTER = 0 
                    face_detection_start_timer = time.time()
                    recalibrating_blinks_start_timer = time.time_ns() / 1000
                    recalibrating_between_blinks_start_timer = time.time_ns() / 1000000
                    print ("\n\n\nError: face is not detected\n\n\n")
                    continue
                
            # calculating FPS
            end_time = time.time() - fps_start_time
            fps = FRAME_COUNTER / end_time
            # frame = utils.textWithBackground(frame, f'FPS: {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            # cv.imshow('Tracking', frame)
            key = cv.waitKey(2)
            # if key==ord('q') or key ==ord('Q'):
            #     break
            
    # cv.destroyAllWindows()
    camera.release()
    # sys.exit()