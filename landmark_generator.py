import copy
import os
import sys
import argparse
import traceback
import gc
import dshowcapture
from math import hypot
'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
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
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
'''

max_threads = 1
os.environ["OMP_NUM_THREADS"] = str(max_threads)

def search_camera(list_cameras = 0, list_dcaps = -1):
    if os.name == 'nt' and (list_cameras > 0 or not list_dcaps is None):
        cap = dshowcapture.DShowCapture()
        info = cap.get_info()
        unit = 10000000.;
        if not list_dcaps is None:
            formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
            for cam in info:
                if list_dcaps == -1:
                    type = ""
                    if cam['type'] == "Blackmagic":
                        type = "Blackmagic: "
                    print(f"{cam['index']}: {type}{cam['name']}")
                if list_dcaps != -1 and list_dcaps != cam['index']:
                    continue
                for caps in cam['caps']:
                    format = caps['format']
                    if caps['format'] in formats:
                        format = formats[caps['format']]
                    if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                        print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                    else:
                        print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
        else:
            if list_cameras == 1:
                print("Available cameras:")
            for cam in info:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                if list_cameras == 1:
                    print(f"{cam['index']}: {type}{cam['name']}")
                else:
                    print(f"{type}{cam['name']}")
        cap.destroy_capture()
    else:
        print("윈도우가 아니라서 실행할 수 없습니다.")

import numpy as np
import time
import cv2
import struct
import json
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path


def run(fps=15, visualize = 0, dcap=None, use_dshowcapture=1, capture="0", log_data="",raw_rgb=0, width=640, height=360, video_out = None, face_id_offset = 0, video_scale=1, threshold=None, max_threads=max_threads, faces=1, discard_after=10, scan_every=3, silent=0, model=3, model_dir=None, gaze_tracking=1, detection_threshold=0.6, scan_retinaface=0, max_feature_updates=900, no_3d_adapt=1, try_hard=0, video_fps = 24, dump_points = ""):
    
    use_dshowcapture_flag = False
    if os.name == 'nt':
        use_dshowcapture_flag = True if use_dshowcapture == 1 else False
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
        if dcap == -1 and type(input_reader) == DShowCaptureReader:
            fps = min(fps, input_reader.device.get_fps())
    else:
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag)
    #if type(input_reader.reader) == VideoReader:
    #    fps = 0
        
    log = None
    out = None
    first = True
    fheight = 0
    fwidth = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    sleep_check = 0

    features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

    if log_data != "":
        log = open(log_data, "w")
        log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
        for i in range(66):
            log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
        for i in range(66):
            log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
        for feature in features:
            log.write(f",{feature}")
        log.write("\r\n")
        log.flush()

    is_camera = capture == str(try_int(capture))

    try:
        attempt = 0
        frame_time = time.perf_counter()
        target_duration = 0
        if fps > 0:
            target_duration = 1. / float(fps)
        need_reinit = 0
        failures = 0
        source_name = input_reader.name
        A_frame = np.empty((0, 68), dtype=int)
        while input_reader.is_open():
            if not input_reader.is_open() or need_reinit == 1:
                input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
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
            if not ret:
                if is_camera:
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
                fheight, fwidth, channels = frame.shape
                tracker = Tracker(fwidth, fheight, threshold=threshold, max_threads=max_threads, max_faces=faces, discard_after=discard_after, scan_every=scan_every, silent=False if silent == 0 else True, model_type=model, model_dir=model_dir, no_gaze=False if gaze_tracking != 0 and model != -1 else True, detection_threshold=detection_threshold, use_retinaface=scan_retinaface, max_feature_updates=max_feature_updates, static_model=True if no_3d_adapt == 1 else False, try_hard=try_hard == 1)
                if not video_out is None:
                    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('F','F','V','1'), video_fps, (fwidth * video_scale, fheight * video_scale))

            try:
                inference_start = time.perf_counter()
                faces = tracker.predict(frame)
                if len(faces) > 0:
                    inference_time = (time.perf_counter() - inference_start)
                    total_tracking_time += inference_time
                    tracking_time += inference_time / len(faces)
                    tracking_frames += 1
                detected = False
                landmarks = np.array([], int) # landmarks in a frame
                for face_num, f in enumerate(faces):
                    f = copy.copy(f)
                    f.id += face_id_offset
                    if f.eye_blink is None:
                        f.eye_blink = [1, 1]
                    right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                    left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                    detected = True
                    if not log is None:
                        log.write(f"{frame_count},{now},{fwidth},{fheight},{fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")

                    for pt_num, (x,y,c) in enumerate(f.lms):
                        if not log is None:
                            log.write(f",{y},{x},{c}")
                        if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                            continue
                        if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                            continue
                        y = int(y + 0.5)
                        
                        landmarks = np.append(landmarks, [x/fheight], axis=0)
                        if visualize != 0 or not out is None:
                            color = (0, 255, 0)
                            if pt_num >= 66:
                                color = (255, 255, 0)
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            x += 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            y += 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            x -= 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                                
                    if f.current_features is None:
                        f.current_features = {}
                    for feature in features:
                        if not feature in f.current_features:
                            f.current_features[feature] = 0
                        if not log is None:
                            log.write(f",{f.current_features[feature]}")
                    if not log is None:
                        log.write("\r\n")
                        log.flush()

                
                if landmarks.size != 68:
                    
                    landmarks = np.append(landmarks, np.zeros(68-landmarks.size), axis=0)
                A_frame = np.vstack([A_frame, landmarks])
                
                if A_frame.size / 2040 == 1:
                    yield A_frame
                    A_frame = np.empty((0, 68), dtype=int)

                if not out is None:
                    video_frame = frame
                    if video_scale != 1:
                        video_frame = cv2.resize(frame, (fwidth * video_scale, fheight * video_scale), interpolation=cv2.INTER_NEAREST)
                    out.write(video_frame)
                    if video_scale != 1:
                        del video_frame

                if visualize != 0:
                    cv2.imshow('OpenSeeFace Visualization', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                failures = 0

            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    if silent == 0:
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
        if silent == 0:
            print("Quitting")

    input_reader.close()
    if not out is None:
        out.release()
    cv2.destroyAllWindows()

    if silent == 0 and tracking_frames > 0:
        average_tracking_time = 1000 * tracking_time / tracking_frames
        print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
        print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}\nFPS: {tracking_frames/total_tracking_time:.3f}")
    

if __name__ == "__main__":
    frame = run(visualize=1, max_threads=4, capture="video.mp4")
    print(frame, frame.size)
