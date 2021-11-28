import copy
import os
import sys
import argparse
import traceback
import gc
import cv2
from math import hypot

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=1)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)

args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

mouth_points = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
r_eye_points = [42, 43, 44, 45, 46, 47]
l_eye_points = [36, 37, 38, 39, 40, 41]


def midpoint(p1, p2): 
    return int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2) 

def get_face_angle(frame, facial_landmarks): 
    try:
        left_point = (facial_landmarks[36*2], facial_landmarks[36*2+1])
        right_point = (facial_landmarks[45*2], facial_landmarks[45*2+1])
    
        left_toppoint = (facial_landmarks[0*2], facial_landmarks[0*2+1])
        right_toppoint = (facial_landmarks[16*2], facial_landmarks[16*2+1])
    
        left_eyebrow = (facial_landmarks[17*2], facial_landmarks[17*2+1])
        right_eyebrow = (facial_landmarks[26*2], facial_landmarks[26*2+1])
    
        left_len = left_point[0] - left_toppoint[0] #왼쪽눈끝점 - 왼쪽얼굴끝점
        right_len = right_toppoint[0] - right_point[0] #오른족 얼굴끝점- 오른쪽눈끝점

    except:
        print("face is not detected")
        return False
    
    #시연용 코드
    cv2.line(frame, left_eyebrow, right_eyebrow, (255, 0, 0), 2)
    cv2.line(frame, left_toppoint, right_toppoint, (0, 255, 0), 2)
    cv2.circle(frame, left_point, 5, (255, 0, 0), 2)
    cv2.circle(frame, right_point, 5, (255, 0, 0), 2)
    cv2.circle(frame, left_toppoint, 5, (0, 255, 0), 2)
    cv2.circle(frame, right_toppoint, 5, (0, 255, 0), 2)
    
    
    if left_len < 0 or right_len < 0:
        cv2.putText(frame,"yaw over!!",org,font,1,(255,0,255),2)
        print("yaw over!!")
        return True
    if abs(right_point[1] - left_point[1]) / (right_point[0]-left_point[0]) > 0.176: # tan(10도) = 0.176 => 10도이상 넘어가면 감지
        cv2.putText(frame,"roll over!!",org,font,1,(255,255,0),2)
        print("roll over!!")
        return True
    if midpoint(left_eyebrow, right_eyebrow)[1] > midpoint(left_toppoint, right_toppoint)[1]:
        cv2.putText(frame,"pitch over!!",org,font,1,(0,255,255),2)
        print("pitch over!!")
        return True
    
    return False

def get_blinking_ratio(frame, eye_points, facial_landmarks): 
    center_top = midpoint(facial_landmarks[eye_points[1]*2:eye_points[1]*2+2], facial_landmarks[eye_points[2]*2:eye_points[2]*2+2])
    center_bottom = midpoint(facial_landmarks[eye_points[4]*2:eye_points[4]*2+2], facial_landmarks[eye_points[5]*2:eye_points[5]*2+2])
    
    left_point = (facial_landmarks[eye_points[0]*2], facial_landmarks[eye_points[0]*2+1])
    right_point = (facial_landmarks[eye_points[3]*2], facial_landmarks[eye_points[3]*2+1])

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) 
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2) 
    hor_line_lenght = hypot( (left_point[0] - right_point[0]), (left_point[1] - right_point[1])) 
    ver_line_lenght = hypot( (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])) 
    ratio = hor_line_lenght / ver_line_lenght
    if ver_line_lenght != 0: 
        ratio = ver_line_lenght / hor_line_lenght
    else: 
        ratio = 60 
    return ratio 

def eye_checker(frame, l_eye_points, r_eye_points, facial_landmarks, sleep_check):
    try:
        left_eye_ratio = get_blinking_ratio(frame, l_eye_points, facial_landmarks)
        right_eye_ratio = get_blinking_ratio(frame, r_eye_points, facial_landmarks)
    
        if (left_eye_ratio + right_eye_ratio) / 2 < 0.18:
            return sleep_check+1, (left_eye_ratio + right_eye_ratio) / 2
        else:
            return 0 , (left_eye_ratio + right_eye_ratio) / 2

    except:
        return 0, 0

max_threads = 1
os.environ["OMP_NUM_THREADS"] = str(max_threads)

import numpy as np
import time
import socket
from input_reader import InputReader, VideoReader, try_int
from tracker import Tracker, get_model_base_path
org=(50,75) 
font=cv2.FONT_HERSHEY_SIMPLEX   

if args.benchmark > 0:
    model_base_path = get_model_base_path(args.model_dir)
    im = cv2.imread(os.path.join(model_base_path, "benchmark.bin"), cv2.IMREAD_COLOR)
    results = []
    for model_type in [3, 2, 1, 0, -1, -2, -3]:
        tracker = Tracker(224, 224, threshold=0.1, max_threads=args.max_threads, max_faces=1, discard_after=0, scan_every=0, silent=True, model_type=model_type, model_dir=args.model_dir, no_gaze=(model_type == -1), detection_threshold=0.1, use_retinaface=0, max_feature_updates=900, static_model=True if args.no_3d_adapt == 1 else False)
        tracker.detected = 1
        tracker.faces = [(0, 0, 224, 224)]
        total = 0.0
        for i in range(100):
            start = time.perf_counter()
            r = tracker.predict(im)
            total += time.perf_counter() - start
        print(1. / (total / 100.))
    sys.exit(0)

target_ip = args.ip
target_port = args.port

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = 24
dcap = None
use_dshowcapture_flag = False

input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)
if type(input_reader.reader) == VideoReader:
    fps = 0

out = None
first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0
sleep_check = 0
plotdata = [[], []]
plotx = 0


features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]


is_camera = args.capture == str(try_int(args.capture))

try:
    startTime = time.time()
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    need_reinit = 0
    failures = 0
    source_name = input_reader.name
    A_frame = np.empty((0, 136), dtype=int)
    while repeat or input_reader.is_open():
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.02)
            continue
        if not input_reader.is_ready():
            time.sleep(0.02)
            continue

        ret, frame = input_reader.read()
        if ret and args.mirror_input:
            frame = cv2.flip(frame, 1)
        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.02)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
            if not args.video_out is None:
                out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('F','F','V','1'), args.video_fps, (width * args.video_scale, height * args.video_scale))

        try:
            inference_start = time.perf_counter()
            faces = tracker.predict(frame)
            if len(faces) > 0:
                inference_time = (time.perf_counter() - inference_start)
                total_tracking_time += inference_time
                tracking_time += inference_time / len(faces)
                tracking_frames += 1

            landmarks = np.array([], int) # landmarks in a frame
            for _, f in enumerate(faces):
                f = copy.copy(f)
                f.id += args.face_id_offset
                
                for pt_num, (x,y,c) in enumerate(f.lms):
                    
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    landmarks = np.append(landmarks, [y, x], axis=0)
                    if args.visualize != 0 or not out is None:
                        if args.visualize > 3:
                            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                        color = (0, 255, 0)
                        if pt_num >= 66:
                            color = (255, 255, 0)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = color
                if args.pnp_points != 0 and (args.visualize != 0 or not out is None) and f.rotation is not None:
                    if args.pnp_points > 1:
                        projected = cv2.projectPoints(f.face_3d[0:66], f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    else:
                        projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    for [(x,y)] in projected[0]:
                        x = int(x + 0.5)
                        y = int(y + 0.5)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)

                if f.current_features is None:
                    f.current_features = {}
                for feature in features:
                    if not feature in f.current_features:
                        f.current_features[feature] = 0

            if not out is None:
                video_frame = frame
                if args.video_scale != 1:
                    video_frame = cv2.resize(frame, (width * args.video_scale, height * args.video_scale), interpolation=cv2.INTER_NEAREST)
                out.write(video_frame)
                if args.video_scale != 1:
                    del video_frame

            if args.visualize != 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if args.dump_points != "" and not faces is None and len(faces) > 0:
                        np.set_printoptions(threshold=sys.maxsize, precision=15)
                        pairs = [
                            (0, 16),
                            (1, 15),
                            (2, 14),
                            (3, 13),
                            (4, 12),
                            (5, 11),
                            (6, 10),
                            (7, 9),
                            (17, 26),
                            (18, 25),
                            (19, 24),
                            (20, 23),
                            (21, 22),
                            (31, 35),
                            (32, 34),
                            (36, 45),
                            (37, 44),
                            (38, 43),
                            (39, 42),
                            (40, 47),
                            (41, 46),
                            (48, 52),
                            (49, 51),
                            (56, 54),
                            (57, 53),
                            (58, 62),
                            (59, 61),
                            (65, 63)
                        ]
                        points = copy.copy(faces[0].face_3d)
                        for a, b in pairs:
                            x = (points[a, 0] - points[b, 0]) / 2.0
                            y = (points[a, 1] + points[b, 1]) / 2.0
                            z = (points[a, 2] + points[b, 2]) / 2.0
                            points[a, 0] = x
                            points[b, 0] = -x
                            points[[a, b], 1] = y
                            points[[a, b], 2] = z
                        points[[8, 27, 28, 29, 33, 50, 55, 60, 64], 0] = 0.0
                        points[30, :] = 0.0
                        with open(args.dump_points, "w") as fh:
                            fh.write(repr(points))
                    break
            failures = 0

            if get_face_angle(frame, landmarks):
                head_check = fps
            else:
                head_check = fps * 2
            plotx += 1
            sleep_check, ploty = eye_checker(frame, l_eye_points, r_eye_points, landmarks, sleep_check)
            plotdata[0].append(plotx)
            plotdata[1].append(ploty)
                
                
            if sleep_check > head_check:
                cv2.putText(frame,"Wake up!!",(50, 50),font,1,(255,0,0),2)
                print("Wake UP!!")

            if landmarks.size != 136:
                    landmarks = np.append(landmarks, np.zeros(136-landmarks.size), axis=0)
            A_frame = np.vstack([A_frame, landmarks])

        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break

        collected = False
        del frame

        duration = time.perf_counter() - frame_time
        while duration < target_duration:
            if not collected:
                gc.collect()
                collected = True
            duration = time.perf_counter() - frame_time
            sleep_time = target_duration - duration
            if sleep_time > 0:                
                time.sleep(sleep_time)
            duration = time.perf_counter() - frame_time
        frame_time = time.perf_counter()
except KeyboardInterrupt:
    if args.silent == 0:
        endTime = time.time()
        print("Quitting")

input_reader.close()
if not out is None:
    out.release()
cv2.destroyAllWindows()

if args.silent == 0 and tracking_frames > 0:
    average_tracking_time = 1000 * tracking_time / tracking_frames
    print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
    print(f"Tracking time: {(endTime-startTime):.3f} s\nFrames: {tracking_frames}\nFPS: {tracking_frames/(endTime-startTime):.3f}")
