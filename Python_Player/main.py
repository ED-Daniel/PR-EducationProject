import cv2 as cv
import mediapipe as mp
import time
import utils, calibrating
import numpy as np
import vg

# global fields
lips_width = 1
eyebrow_height_r = 1
eyebrow_height_l = 1
blinks = 1
average_blink_dur = 1

# variables 
FRAME_COUNTER = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
BLINKS_IN_MINUTE = 0
HEAD_FLUCTUATION_IN_TIME = 0
HEAD_FLUCTUATION_COUNTER = 0
HEAD_CHECK_TIMER = 30
CALIB_TIME = 240
CALIBRATION_DURATION = 30

# constants
CLOSED_EYES_FRAME = 1
FONTS = cv.FONT_HERSHEY_COMPLEX

#const
KWIDTH = 1.2
KHEIGHT = 1.65
TRACK_CONF = 0.7
DETECT_CONF = 0.7

# face bounder indices 
FACE_OVAL = [ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109 ]

# lips indices for Landmarks
LIPS = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS = [ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95 ]
UPPER_LIPS = [ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ] 

# Left eyes indices 
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
LEFT_EYEBROW = [ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
LEFT_IRIS = [ 474, 475, 476, 477 ]

# Right eyes indices
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]  
RIGHT_EYEBROW = [ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
RIGHT_IRIS = [ 469, 470, 471, 472 ]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv.VideoCapture(0)

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

#attention tracking
def lostAttention(angleOX, angleOZ, point_ir, point_center, rad):
    normalizedAngleOX = abs(90 - angleOX)
    #normalizedAngleOZ = abs(90 - angleOZ)
    flag = False
    if (point_ir[0] - point_center[0])**2 + (point_ir[1] - point_center[1])**2 <= rad**2: flag = True
    if normalizedAngleOX > 30 or flag is False:
        return True
    return False

#tiredness counter
def findTiredRatio(bl, dur):
    norm_bl = bl - blinks
    if norm_bl < 0: norm_bl = 0
    norm_dur = dur - average_blink_dur
    if norm_dur < 0: norm_dur = 0
    ratio = (norm_bl * 50 / blinks) + (norm_dur * 50 / average_blink_dur)
    if ratio > 100:
        ratio = 100
    return ratio

#happiness counter
def smileCounter(landmarks):
    minLength = lips_width
    maxLength = lips_width * 0.2
    leftCorner = landmarks[LIPS[0]][:2] # 61
    rightCorner = landmarks[LIPS[10]][:2] # 291
    #upCenter = landmarks[LIPS[25]][:2] # 0
    #downCenter = landmarks[LIPS[5]][:2] # 17
    lengthOfSmile = utils.euclaideanDistance2D(leftCorner, rightCorner)
    percent = round((lengthOfSmile - minLength) / maxLength * 100)
    if percent < 5: percent = 0
    elif percent > 100: percent = 100
    return percent
    
#amazement counter
def amazeCounter(dr, dl):
    minLength = (eyebrow_height_l + eyebrow_height_r) / 2
    max_h = minLength * 0.045
    percent = round((((dr + dl) / 2) - minLength) / max_h * 100)
    if percent > 100: percent = 100
    elif percent < 5: percent = 0
    return percent

with map_face_mesh.FaceMesh(min_detection_confidence = DETECT_CONF, min_tracking_confidence = TRACK_CONF, refine_landmarks = True) as face_mesh:
    # starting time here 
    blink_counting_start_time = time.time()
    start_time = time.time()
    head_fluctuation_counting_starting_time = time.time()
    recalibrating_starting_time = time.time()
    recalibrate_start = 0
    
    #calibration
    flag_calib, lips_width, eyebrow_height_l, eyebrow_height_r, blinks, average_blink_dur = calibrating.Calibrate(camera, face_mesh, True)
    
    if flag_calib is True:
        LAST_DURATION = 0
        AV_BLINK_DURATION = []
        av_dur = average_blink_dur
        tiredRatio = 0
        
        #recalib.
        CALIBRATING_SMILE_POSITIONS = []
        CALIBRATING_EYEBROWS_POSITIONS = []
        CALIBRATING_BLINKS_DURATION = []
        BLINKS_COUNTER2 = 0
        CEF_COUNTER2 = 0
        start_time2 = time.time()

        while True:
            FRAME_COUNTER +=1 # frame counter
            ret, frame = camera.read() # getting frame from camera 
            if not ret: 
                break # no more frames break
            
            #  resizing frame
            frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                mesh_coords, mesh_coords_z = landmarksDetection(frame, results)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
                if CEF_COUNTER == 1: timer_started = time.time_ns() / 1000
                if ratio > 5.5:
                    CEF_COUNTER +=1
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, 6, 6)
                elif CEF_COUNTER>=CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    LAST_DURATION = time.time_ns() / 1000 - timer_started
                    #fale blink are not counted
                    if LAST_DURATION < 200000 : AV_BLINK_DURATION.append(LAST_DURATION)
                    CEF_COUNTER =0
                utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                utils.colorBackgroundText(frame,  f'Last blink duration mcsec: {round(LAST_DURATION, 2)}', FONTS, 0.7, (200,100),2)
                
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
                
                #head angles
                p1 = (0, int(frame_height / 2))
                p2 = (int(frame_width), int(frame_height / 2))
                vec3 = np.array([0, 0, 1000])
                #angle between horizontal and face vertical lines
                vec1 = np.array([p2[0] - p1[0], p2[1] - p1[1], 0])
                vec2 = np.array([mesh_coords[175][0] - mesh_coords[151][0], mesh_coords[175][1] - mesh_coords[151][1], mesh_coords_z[175][2] - mesh_coords_z[151][2]])
                angle_h = vg.angle(vec1, vec2, look=vg.basis.z)
                angle_v = vg.angle(vec2, vec3, look=vg.basis.y)
                #showing head angles
                #utils.colorBackgroundText(frame, f'Hor_Angle: {round(angle_h, 0)}', FONTS, 1.0, (frame_width - 250, 250), 2, (0,255,0), utils.RED, 8, 8)
                #utils.colorBackgroundText(frame, f'Ver_Angle: {round(angle_v, 0)}', FONTS, 1.0, (frame_width - 250, 250), 2, (0,255,0), utils.RED, 8, 8)
                
                #irises
                r_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
                l_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
                
                if ratio < 5.5:
                    r = utils.seg_intersect(np.array(r_iris_coords[0]), np.array(r_iris_coords[2]), np.array(r_iris_coords[1]), np.array(r_iris_coords[3]))
                    r_iris_center = [round(r[0]), round(r[1])]
                    l = utils.seg_intersect(np.array(l_iris_coords[0]), np.array(l_iris_coords[2]), np.array(l_iris_coords[1]), np.array(l_iris_coords[3]))
                    l_iris_center = [round(l[0]), round(l[1])]
                    try:
                        c_r = utils.seg_intersect(np.array(mesh_coords[133]), np.array(mesh_coords[33]), np.array(mesh_coords[159]), np.array(mesh_coords[153]))
                        center_r = [round(c_r[0]), round(c_r[1])]
                        c_l = utils.seg_intersect(np.array(mesh_coords[362]), np.array(mesh_coords[263]), np.array(mesh_coords[380]), np.array(mesh_coords[386]))
                        center_l = [round(c_l[0]), round(c_l[1])]
                        right_iris = [r_iris_center[0] - center_r[0], r_iris_center[1] - center_r[1]]
                        left_iris = [l_iris_center[0] - center_l[0], l_iris_center[1] - center_l[1]]
                        r_size = [utils.euclaideanDistance3D(mesh_coords_z[173], mesh_coords_z[33]), utils.euclaideanDistance3D(mesh_coords_z[159], mesh_coords_z[153])]
                        l_size = [utils.euclaideanDistance3D(mesh_coords_z[398], mesh_coords_z[263]), utils.euclaideanDistance3D(mesh_coords_z[380], mesh_coords_z[386])]
                        r_prop_k = [round(frame_width / r_size[0]), round(frame_height / r_size[1])]
                        l_prop_k = [round(frame_width / l_size[0]), round(frame_height / l_size[1])]
                        center_screen = [round(frame_width / 2.), round(frame_height / 2.)]
                        cv.circle(frame, center = (round(center_screen[0]), round(center_screen[1])), radius = round(frame_height / 2 * 0.8), color = utils.RED, thickness=2)
                        #glance position on screen
                        screen_iris_l = [center_screen[0] - (left_iris[0] * l_prop_k[0] * KWIDTH), center_screen[1] + (left_iris[1] * l_prop_k[1] * KHEIGHT)]
                        screen_iris_r = [center_screen[0] - (right_iris[0] * r_prop_k[0] * KWIDTH), center_screen[1] + (right_iris[1] * r_prop_k[1] * KHEIGHT)]
                        screen_irises = [round((screen_iris_l[0] + screen_iris_r[0]) / 2), round((screen_iris_l[1] + screen_iris_r[1]) / 2)]
                        if utils.euclaideanDistance2D(screen_iris_l, screen_iris_r) < frame_width/2:
                            cv.circle(frame, center = (round(screen_irises[0]), round(screen_irises[1])), radius = 50, color = utils.RED, thickness=2)
                        #iris position
      # =============================================================================
      #                 utils.colorBackgroundText(frame, f'R_i pos: {right_iris}', FONTS, 1.0, (40, 270), 2, (0,255,0), utils.RED, 8, 8)
      #                 utils.colorBackgroundText(frame, f'L_i pos: {left_iris}', FONTS, 1.0, (40, 300), 2, (0,255,0), utils.RED, 8, 8)
      # =============================================================================
                    except:
                        print("iris tracking error")    
                        
                #attention
                if lostAttention(angle_h, angle_v, screen_irises, center_screen, (frame_height / 2 * 0.8)) is True:
                    utils.colorBackgroundText(frame, 'LOST ATTENTION!', FONTS, 1.7, (int(frame_height/2), 600), 2, utils.YELLOW, pad_x=6, pad_y=6)
                    
                #happiness
                smile = smileCounter(mesh_coords_z)
                utils.colorBackgroundText(frame, f'Happiness: {smile}', FONTS, 1.0, (frame_width - 250, 50), 2, (0,255,0), utils.RED, 8, 8)
                
                #amaze
                amaze = amazeCounter(utils.euclaideanDistance2D(mesh_coords[4], mesh_coords[193]), utils.euclaideanDistance2D(mesh_coords[4], mesh_coords[417]))
                utils.colorBackgroundText(frame, f'Amaze: {amaze}', FONTS, 1.0, (frame_width - 250, 100), 2, (0,255,0), utils.RED, 8, 8)
                
                #tiredness
                if time.time() - blink_counting_start_time >= 60:
                    if len(AV_BLINK_DURATION) is 0: av_dur = average_blink_dur
                    av_dur = sum([i for i in AV_BLINK_DURATION]) / len(AV_BLINK_DURATION)
                    BLINKS_IN_MINUTE = TOTAL_BLINKS - BLINKS_IN_MINUTE
                    blink_counting_start_time = time.time()
                    AV_BLINK_DURATION.clear()
                    tiredRatio = findTiredRatio(BLINKS_IN_MINUTE, av_dur)
                    #utils.colorBackgroundText(frame, f'Blinks in minute: {BLINKS_IN_MINUTE}', FONTS, 1.0, (30,200), 2, (0,255,0), utils.RED, 8, 8)
                utils.colorBackgroundText(frame, f'Tiredness: {tiredRatio}%', FONTS, 1.0, (frame_width - 250, 200), 2, (0,255,0), utils.RED, 8, 8)

                #recalibrating
                # if time.time() - recalibrating_starting_time >= CALIB_TIME and flag_calib is False:
                #     flag_calib = False
                #     # Calibrating happiness
                #     leftCorner = mesh_coords_z[LIPS[0]]
                #     rightCorner = mesh_coords_z[LIPS[10]]
                #     CALIBRATING_SMILE_POSITIONS.append(tuple([leftCorner, rightCorner]))
                        
                #     # Calibrating amaze
                #     temp_eyebrow_height_r = utils.euclaideanDistance2D(mesh_coords[4], mesh_coords[193])
                #     temp_eyebrow_height_l = utils.euclaideanDistance2D(mesh_coords[4], mesh_coords[417])
                #     CALIBRATING_EYEBROWS_POSITIONS.append(tuple([temp_eyebrow_height_l, temp_eyebrow_height_r]))
                        
                #     # Calibrating tiredness
                #     if CEF_COUNTER2 == 1: timer_started = time.time_ns() / 1000
                #     if ratio2 > 5.5:
                #         CEF_COUNTER2 +=1
                #     else:
                #         if CEF_COUNTER2 >= CLOSED_EYES_FRAME:
                #             BLINKS_COUNTER2 +=1
                #             CALIBRATING_BLINKS_DURATION.append(time.time_ns() / 1000 - timer_started)
                #             timer_started = time.time_ns() / 1000
                #             CEF_COUNTER2 =0
                        
                #         #average
                #         if len(CALIBRATING_SMILE_POSITIONS) > 0:
                #                     lips_width = sum([utils.euclaideanDistance2D(i[0][:2], i[1][:2]) for i in CALIBRATING_SMILE_POSITIONS]) / len(CALIBRATING_SMILE_POSITIONS)
                #                     flag_calib = True
                #         else:
                #                     flag_calib = False
                #         if len(CALIBRATING_EYEBROWS_POSITIONS) > 0:
                #                     eyebrow_height_l = sum([i[0] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)
                #                     eyebrow_height_r = sum([i[1] for i in CALIBRATING_EYEBROWS_POSITIONS]) / len(CALIBRATING_EYEBROWS_POSITIONS)
                #                     flag_calib = True
                #         else:
                #                     flag_calib = False
                #         if CALIBRATION_DURATION > 0:
                #                     blinks = (BLINKS_COUNTER / CALIBRATION_DURATION) * (60.0 / CALIBRATION_DURATION)
                #                     flag_calib = True
                #         else: 
                #                     flag_calib = False
                #         if len(CALIBRATING_BLINKS_DURATION) > 0:
                #                     average_blink_dur = sum([i for i in CALIBRATING_BLINKS_DURATION]) / len(CALIBRATING_BLINKS_DURATION)
                #                     if average_blink_dur <= 0: average_blink_dur = 1
                #                     flag_calib = True
                #         else:
                #                     flag_calib = False
                                
                #                 #printing results
                #         if flag_calib is True:
                #             print("NEW CALIBRATING RESULTS: ")
                #             print("NEW AVERAGE MOUTH LENGTH: ", lips_width)
                #             print("NEW AVERAGE LEFT EYEBROW HEIGHT: ", eyebrow_height_l)
                #             print("NEW AVERAGE RIGHT EYEBROW HEIGHT: ", eyebrow_height_r)
                #             print("NEW AVERAGE BLINKS PER MINUTE: ", blinks)
                #             print("NEW AVERAGE BLINK DURATION: ", average_blink_dur)
                    
                #     remaining_time = CALIBRATION_DURATION - (time.time()-start_time)
                #     if remaining_time <= 0:
                #         flag_calib = True
                #     recalibrating_starting_time = time.time()
                    
                #head fulctuation
                if abs(90 - angle_h) > 30:
                    HEAD_FLUCTUATION_COUNTER += 1
                if time.time() - head_fluctuation_counting_starting_time >= HEAD_CHECK_TIMER:
                    HEAD_FLUCTUATION_IN_TIME = HEAD_FLUCTUATION_COUNTER
                    head_fluctuation_counting_starting_time = time.time()
                    HEAD_FLUCTUATION_COUNTER = 0
                utils.colorBackgroundText(frame, f'Head fluctuation per 30s: {HEAD_FLUCTUATION_IN_TIME}%', FONTS, 1.0, (10, 250), 2, (0,255,0), utils.RED, 8, 8)
                
            # calculating frame per seconds FPS
            end_time = time.time()-start_time
            fps = FRAME_COUNTER/end_time
            frame = utils.textWithBackground(frame, f'FPS: {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            cv.imshow('Tracking', frame)
            key = cv.waitKey(2)
            if key==ord('q') or key ==ord('Q'):
                break
        else:
            print ("face is not detected")
            
    cv.destroyAllWindows()
    camera.release()
