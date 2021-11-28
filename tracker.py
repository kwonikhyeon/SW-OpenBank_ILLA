import os
import numpy as np
import math
import cv2
import onnxruntime
import time
import queue
import threading
from retinaface import RetinaFaceDetector
from remedian import remedian

def resolve(name):
    f = os.path.join(os.path.dirname(__file__), name)
    return f

def clamp_to_im(pt, w, h):
    x = pt[0]
    y = pt[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= w:
        x = w-1
    if y >= h:
        y = h-1
    return (int(x), int(y+1))

def rotate(origin, point, a):
    a = -a
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy

def angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.arctan2(*(p2 - p1)[::-1])
    return (a % (2 * np.pi))

def compensate(p1, p2):
    a = angle(p1, p2)
    return rotate(p1, p2, a), a

def rotate_image(image, a, center):
    (h, w) = image.shape[:2]
    a = np.rad2deg(a)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), a, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def intersects(r1, r2, amount=0.3):
    area1 = r1[2] * r1[3]
    area2 = r2[2] * r2[3]
    inter = 0.0
    total = area1 + area2
    
    r1_x1, r1_y1, w, h = r1
    r1_x2 = r1_x1 + w
    r1_y2 = r1_y1 + h
    r2_x1, r2_y1, w, h = r2
    r2_x2 = r2_x1 + w
    r2_y2 = r2_y1 + h

    left = max(r1_x1, r2_x1)
    right = min(r1_x2, r2_x2)
    top = max(r1_y1, r2_y1)
    bottom = min(r1_y2, r2_y2)
    if left < right and top < bottom:
        inter = (right - left) * (bottom - top)
        total -= inter

    if inter / total >= amount:
        return True

    return False

def group_rects(rects):
    rect_groups = {}
    for rect in rects:
        rect_groups[str(rect)] = [-1, -1, []]
    group_id = 0
    for i, rect in enumerate(rects):
        name = str(rect)
        group = group_id
        group_id += 1
        if rect_groups[name][0] < 0:
            rect_groups[name] = [group, -1, []]
        else:
            group = rect_groups[name][0]
        for j, other_rect in enumerate(rects):
            if i == j:
                continue;
            if intersects(rect, other_rect):
                rect_groups[str(other_rect)] = [group, -1, []]
    return rect_groups

def logit(p, factor=16.0):
    if p >= 1.0:
        p = 0.9999999
    if p <= 0.0:
        p = 0.0000001
    p = p/(1-p)
    return float(np.log(p)) / float(factor)

def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)

def worker_thread(session, frame, input, crop_info, queue, input_name, idx, tracker):
    output = session.run([], {input_name: input})[0]
    conf, lms = tracker.landmarks(output[0], crop_info)
    if conf > tracker.threshold:
        try:
            eye_state = tracker.get_eye_state(frame, lms)
        except:
            eye_state = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        queue.put((session, conf, (lms, eye_state), crop_info, idx))
    else:
        queue.put((session,))

class Feature():
    def __init__(self, threshold=0.15, alpha=0.2, hard_factor=0.15, decay=0.001, max_feature_updates=0):
        self.median = remedian()
        self.min = None
        self.max = None
        self.hard_min = None
        self.hard_max = None
        self.threshold = threshold
        self.alpha = alpha
        self.hard_factor = hard_factor
        self.decay = decay
        self.last = 0
        self.current_median = 0
        self.update_count = 0
        self.max_feature_updates = max_feature_updates
        self.first_seen = -1
        self.updating = True

    def update(self, x, now=0):
        if self.max_feature_updates > 0:
            if self.first_seen == -1:
                self.first_seen = now;
        new = self.update_state(x, now=now)
        filtered = self.last * self.alpha + new * (1 - self.alpha)
        self.last = filtered
        return filtered

    def update_state(self, x, now=0):
        updating = self.updating and (self.max_feature_updates == 0 or now - self.first_seen < self.max_feature_updates)
        if updating:
            self.median + x
            self.current_median = self.median.median()
        else:
            self.updating = False
        median = self.current_median

        if self.min is None:
            if x < median and (median - x) / median > self.threshold:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
            return 0
        else:
            if x < self.min:
                if updating:
                    self.min = x
                    self.hard_min = self.min + self.hard_factor * (median - self.min)
                return -1
        if self.max is None:
            if x > median and (x - median) / median > self.threshold:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1
            return 0
        else:
            if x > self.max:
                if updating:
                    self.max = x
                    self.hard_max = self.max - self.hard_factor * (self.max - median)
                return 1

        if updating:
            if self.min < self.hard_min:
                self.min = self.hard_min * self.decay + self.min * (1 - self.decay)
            if self.max > self.hard_max:
                self.max = self.hard_max * self.decay + self.max * (1 - self.decay)

        if x < median:
            return - (1 - (x - self.min) / (median - self.min))
        elif x > median:
            return (x - median) / (self.max - median)

        return 0

class FeatureExtractor():
    def __init__(self, max_feature_updates=0):
        self.eye_l = Feature(max_feature_updates=max_feature_updates)
        self.eye_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_quirk_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_l = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.eyebrow_steepness_r = Feature(threshold=0.05, max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_l = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_updown_r = Feature(max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_l = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_corner_inout_r = Feature(threshold=0.02, max_feature_updates=max_feature_updates)
        self.mouth_open = Feature(max_feature_updates=max_feature_updates)
        self.mouth_wide = Feature(threshold=0.02, max_feature_updates=max_feature_updates)

    def align_points(self, a, b, pts):
        a = tuple(a)
        b = tuple(b)
        alpha = angle(a, b)
        alpha = np.rad2deg(alpha)
        if alpha >= 90:
            alpha = - (alpha - 180)
        if alpha <= -90:
            alpha = - (alpha + 180)
        alpha = np.deg2rad(alpha)
        aligned_pts = []
        for pt in pts:
            aligned_pts.append(np.array(rotate(a, pt, alpha)))
        return alpha, np.array(aligned_pts)

    def update(self, pts, full=True):
        features = {}
        now = time.perf_counter()

        norm_distance_x = np.mean([pts[0, 0] - pts[16, 0], pts[1, 0] - pts[15, 0]])
        norm_distance_y = np.mean([pts[27, 1] - pts[28, 1], pts[28, 1] - pts[29, 1], pts[29, 1] - pts[30, 1]])

        a1, f_pts = self.align_points(pts[42], pts[45], pts[[43, 44, 47, 46]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_l"] = self.eye_l.update(f, now)

        a2, f_pts = self.align_points(pts[36], pts[39], pts[[37, 38, 41, 40]])
        f = abs((np.mean([f_pts[0,1], f_pts[1,1]]) - np.mean([f_pts[2,1], f_pts[3,1]])) / norm_distance_y)
        features["eye_r"] = self.eye_r.update(f, now)

        if full:
            a3, _ = self.align_points(pts[0], pts[16], [])
            a4, _ = self.align_points(pts[31], pts[35], [])
            norm_angle = np.mean(list(map(np.rad2deg, [a1, a2, a3, a4])))

            a, f_pts = self.align_points(pts[22], pts[26], pts[[22, 23, 24, 25, 26]])
            features["eyebrow_steepness_l"] = self.eyebrow_steepness_l.update(-np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_l"] = self.eyebrow_quirk_l.update(f, now)

            a, f_pts = self.align_points(pts[17], pts[21], pts[[17, 18, 19, 20, 21]])
            features["eyebrow_steepness_r"] = self.eyebrow_steepness_r.update(np.rad2deg(a) - norm_angle, now)
            f = np.max(np.abs(np.array(f_pts[1:4]) - f_pts[0, 1])) / norm_distance_y
            features["eyebrow_quirk_r"] = self.eyebrow_quirk_r.update(f, now)
        else:
            features["eyebrow_steepness_l"] = 0.
            features["eyebrow_steepness_r"] = 0.
            features["eyebrow_quirk_l"] = 0.
            features["eyebrow_quirk_r"] = 0.

        f = (np.mean([pts[22, 1], pts[26, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_l"] = self.eyebrow_updown_l.update(f, now)

        f = (np.mean([pts[17, 1], pts[21, 1]]) - pts[27, 1]) / norm_distance_y
        features["eyebrow_updown_r"] = self.eyebrow_updown_r.update(f, now)

        upper_mouth_line = np.mean([pts[49, 1], pts[50, 1], pts[51, 1]])
        center_line = np.mean([pts[50, 0], pts[60, 0], pts[27, 0], pts[30, 0], pts[64, 0], pts[55, 0]])

        f = (upper_mouth_line - pts[62, 1]) / norm_distance_y
        features["mouth_corner_updown_l"] = self.mouth_corner_updown_l.update(f, now)
        if full:
            f = abs(center_line - pts[62, 0]) / norm_distance_x
            features["mouth_corner_inout_l"] = self.mouth_corner_inout_l.update(f, now)
        else:
            features["mouth_corner_inout_l"] = 0.

        f = (upper_mouth_line - pts[58, 1]) / norm_distance_y
        features["mouth_corner_updown_r"] = self.mouth_corner_updown_r.update(f, now)
        if full:
            f = abs(center_line - pts[58, 0]) / norm_distance_x
            features["mouth_corner_inout_r"] = self.mouth_corner_inout_r.update(f, now)
        else:
            features["mouth_corner_inout_r"] = 0.

        f = abs(np.mean(pts[[59,60,61], 1], axis=0) - np.mean(pts[[63,64,65], 1], axis=0)) / norm_distance_y
        features["mouth_open"] = self.mouth_open.update(f, now)

        f = abs(pts[58, 0] - pts[62, 0]) / norm_distance_x
        features["mouth_wide"] = self.mouth_wide.update(f, now)

        return features

class FaceInfo():
    def __init__(self, id, tracker):
        self.id = id
        self.frame_count = -1
        self.tracker = tracker
        self.contour_pts = [0,1,8,15,16,27,28,29,30,31,32,33,34,35]
        if self.tracker.model_type == -1:
            self.contour_pts = [0,2,8,14,16,27,30,33]
        self.reset()
        self.alive = False
        self.coord = None

        self.update_count_delta = 75.
        self.update_count_max = 7500.

        if self.tracker.max_feature_updates > 0:
            self.features = FeatureExtractor(self.tracker.max_feature_updates)

    def reset(self):
        self.alive = False
        self.conf = None
        self.lms = None
        self.eye_state = None
        self.rotation = None
        self.translation = None
        self.success = None
        self.quaternion = None
        self.euler = None
        self.pnp_error = None
        self.eye_blink = None
        self.bbox = None
        self.pnp_error = 0
        if self.tracker.max_feature_updates < 1:
            self.features = FeatureExtractor(0)
        self.current_features = {}
        self.contour = np.zeros((21,3))
        self.update_counts = np.zeros((66,2))
        self.fail_count = 0

    def update(self, result, coord, frame_count):
        self.frame_count = frame_count
        if result is None:
            self.reset()
        else:
            self.conf, (self.lms, self.eye_state) = result
            self.coord = coord
            self.alive = True

        if self.tracker.model_type != -1 and not self.tracker.static_model:
            changed_any = False

            if changed_any:
                # Update weighted by point confidence
                weights = np.zeros((66,3))
                weights[:, :] = self.lms[0:66, 2:3]
                weights[weights > 0.7] = 1.0
                weights = 1.0 - weights

def get_model_base_path(model_dir):
    model_base_path = resolve(os.path.join("models"))
    if model_dir is None:
        if not os.path.exists(model_base_path):
            model_base_path = resolve(os.path.join("..", "models"))
    else:
        model_base_path = model_dir
    return model_base_path

class Tracker():
    def __init__(self, width, height, model_type=3, detection_threshold=0.6, threshold=None, max_faces=1, discard_after=5, scan_every=3, bbox_growth=0.0, max_threads=4, silent=False, model_dir=None, no_gaze=False, use_retinaface=False, max_feature_updates=0, static_model=False, feature_level=2, try_hard=False):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = min(max_threads,4)
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.model_type = model_type
        self.models = [
            "lm_model0_opt.onnx",
            "lm_model1_opt.onnx",
            "lm_model2_opt.onnx",
            "lm_model3_opt.onnx",
            "lm_model4_opt.onnx"
        ]
        model = "lm_modelT_opt.onnx"
        if model_type >= 0:
            model = self.models[self.model_type]
        if model_type == -2:
            model = "lm_modelV_opt.onnx"
        if model_type == -3:
            model = "lm_modelU_opt.onnx"
        model_base_path = get_model_base_path(model_dir)

        if threshold is None:
            threshold = 0.6
            if model_type < 0:
                threshold = 0.87

        self.retinaface = RetinaFaceDetector(model_path=os.path.join(model_base_path, "retinaface_640x640_opt.onnx"), json_path=os.path.join(model_base_path, "priorbox_640x640.json"), threads=max(max_threads,4), top_k=max_faces, res=(640, 640))
        self.retinaface_scan = RetinaFaceDetector(model_path=os.path.join(model_base_path, "retinaface_640x640_opt.onnx"), json_path=os.path.join(model_base_path, "priorbox_640x640.json"), threads=2, top_k=max_faces, res=(640, 640))
        self.use_retinaface = use_retinaface

        # Single face instance with multiple threads
        self.session = onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options)

        # Multiple faces with single threads
        self.sessions = []
        self.max_workers = max(min(max_threads, max_faces), 1)
        extra_threads = max_threads % self.max_workers
        for i in range(self.max_workers):
            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = min(max(max_threads // self.max_workers, 4), 1)
            if options.intra_op_num_threads < 1:
                options.intra_op_num_threads = 1
            elif i < extra_threads:
                options.intra_op_num_threads += 1
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions.append(onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options))
        self.input_name = self.session.get_inputs()[0].name

        options = onnxruntime.SessionOptions()
        #options.intra_op_num_threads = max(max_threads,4)
        options.intra_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.gaze_model = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_gaze32_split_opt.onnx"), sess_options=options)

        self.detection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options)
        self.faces = []

        # Image normalization constants
        self.mean = np.float32(np.array([0.485, 0.456, 0.406]))
        self.std = np.float32(np.array([0.229, 0.224, 0.225]))
        self.mean = self.mean / self.std
        self.std = self.std * 255.0

        self.mean = - self.mean
        self.std = 1.0 / self.std
        self.mean_32 = np.tile(self.mean, [32, 32, 1])
        self.std_32 = np.tile(self.std, [32, 32, 1])
        self.mean_224 = np.tile(self.mean, [224, 224, 1])
        self.std_224 = np.tile(self.std, [224, 224, 1])

        self.camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], np.float32)
        self.inverse_camera = np.linalg.inv(self.camera)
        self.dist_coeffs = np.zeros((4,1))

        self.frame_count = 0
        self.width = width
        self.height = height
        self.threshold = threshold
        self.detection_threshold = detection_threshold
        self.max_faces = max_faces
        self.max_threads = max_threads
        self.discard = 0
        self.discard_after = discard_after
        self.detected = 0
        self.wait_count = 0
        self.scan_every = scan_every
        self.bbox_growth = bbox_growth
        self.silent = silent
        self.try_hard = try_hard

        self.res = 224.
        self.mean_res = self.mean_224
        self.std_res = self.std_224
        if model_type < 0:
            self.res = 56.
            self.mean_res = np.tile(self.mean, [56, 56, 1])
            self.std_res = np.tile(self.std, [56, 56, 1])
        if model_type < -1:
            self.res = 112.
            self.mean_res = np.tile(self.mean, [112, 112, 1])
            self.std_res = np.tile(self.std, [112, 112, 1])
        self.res_i = int(self.res)
        self.out_res = 27.
        if model_type < 0:
            self.out_res = 6.
        if model_type < -1:
            self.out_res = 13.
        self.out_res_i = int(self.out_res) + 1
        self.logit_factor = 16.
        if model_type < 0:
            self.logit_factor = 8.
        if model_type < -1:
            self.logit_factor = 16.

        self.no_gaze = no_gaze
        self.debug_gaze = False
        self.feature_level = feature_level
        if model_type == -1:
            self.feature_level = min(feature_level, 1)
        self.max_feature_updates = max_feature_updates
        self.static_model = static_model
        self.face_info = [FaceInfo(id, self) for id in range(max_faces)]
        self.fail_count = 0

    def detect_faces(self, frame):
        im = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[:,:,::-1] * self.std_224 + self.mean_224
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        outputs, maxpool = self.detection.run([], {'input': im})
        outputs = np.array(outputs)
        maxpool = np.array(maxpool)
        outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
        detections = np.flip(np.argsort(outputs[0,0].flatten()))
        results = []
        for det in detections[0:self.max_faces]:
            y, x = det // 56, det % 56
            c = outputs[0, 0, y, x]
            r = outputs[0, 1, y, x] * 112.
            x *= 4
            y *= 4
            r *= 1.0
            if c < self.detection_threshold:
                break
            results.append((x - r, y - r, 2 * r, 2 * r * 1.0))
        results = np.array(results).astype(np.float32)
        if results.shape[0] > 0:
            results[:, [0,2]] *= frame.shape[1] / 224.
            results[:, [1,3]] *= frame.shape[0] / 224.
        return results

    def landmarks(self, tensor, crop_info):
        crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
        avg_conf = 0
        res = self.res - 1
        c0, c1, c2 = 66, 132, 198
        if self.model_type == -1:
            c0, c1, c2 = 30, 60, 90
        t_main = tensor[0:c0].reshape((c0,self.out_res_i * self.out_res_i))
        t_m = t_main.argmax(1)
        indices = np.expand_dims(t_m, 1)
        t_conf = np.take_along_axis(t_main, indices, 1).reshape((c0,))
        t_off_x = np.take_along_axis(tensor[c0:c1].reshape((c0,self.out_res_i * self.out_res_i)), indices, 1).reshape((c0,))
        t_off_y = np.take_along_axis(tensor[c1:c2].reshape((c0,self.out_res_i * self.out_res_i)), indices, 1).reshape((c0,))
        t_off_x = res * logit_arr(t_off_x, self.logit_factor)
        t_off_y = res * logit_arr(t_off_y, self.logit_factor)
        t_x = crop_y1 + scale_y * (res * np.floor(t_m / self.out_res_i) / self.out_res + t_off_x)
        t_y = crop_x1 + scale_x * (res * np.floor(np.mod(t_m, self.out_res_i)) / self.out_res + t_off_y)
        avg_conf = np.average(t_conf)
        lms = np.stack([t_x, t_y, t_conf], 1)
        lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
        if self.model_type == -1:
            lms = lms[[0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,7,7,8,8,9,10,10,11,11,12,21,21,21,22,23,23,23,23,23,13,14,14,15,16,16,17,18,18,19,20,20,24,25,25,25,26,26,27,27,27,24,24,28,28,28,26,29,29,29]]
            #lms[[1,3,4,6,7,9,10,12,13,15,18,20,23,25,38,40,44,46]] += lms[[2,2,5,5,8,8,11,11,14,16,19,21,24,26,39,39,45,45]]
            #lms[[3,4,6,7,9,10,12,13]] += lms[[5,5,8,8,11,11,14,14]]
            #lms[[1,15,18,20,23,25,38,40,44,46]] /= 2.0
            #lms[[3,4,6,7,9,10,12,13]] /= 3.0
            part_avg = np.mean(np.partition(lms[:,2],3)[0:3])
            if part_avg < 0.65:
                avg_conf = part_avg
        return (avg_conf, np.array(lms))

    def preprocess(self, im, crop):
        x1, y1, x2, y2 = crop
        im = np.float32(im[y1:y2, x1:x2,::-1]) # Crop and BGR to RGB
        im = cv2.resize(im, (self.res_i, self.res_i), interpolation=cv2.INTER_LINEAR) * self.std_res + self.mean_res
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        return im

    def equalize(self, im):
        im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])
        return cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    def corners_to_eye(self, corners, w, h):
        ((cx1, cy1), (cx2, cy2)) = corners
        c1 = np.array([cx1, cy1])
        c2 = np.array([cx2, cy2])
        c2, a = compensate(c1, c2)
        center = (c1 + c2) / 2.0
        radius = np.linalg.norm(c1 - c2) / 2.0
        radius = np.array([radius * 1.4, radius * 1.2])
        upper_left = clamp_to_im(center - radius, w, h)
        lower_right = clamp_to_im(center + radius, w, h)
        return upper_left, lower_right, c1, a

    def prepare_eye(self, frame, lms):
        outer_pt = tuple(lms[0])
        inner_pt = tuple(lms[1])
        h, w, _ = frame.shape
        (x1, y1), (x2, y2), reference, a = self.corners_to_eye((outer_pt, inner_pt), w, h)
        im = rotate_image(frame[:, :, ::], a, reference)
        im = im[int(y1):int(y2), int(x1):int(x2),:]
        if np.prod(im.shape) < 1:
            return None, None, None, None, None, None
        #im = self.equalize(im)

        im = im.astype(np.float32)[:,:,::-1] * self.std_32 + self.mean_32
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,2,1))
        return im, x1, y1, reference, a

    def extract_face(self, frame, lms):
        lms = np.array(lms)[:,0:2][:,::-1]
        x1, y1 = tuple(lms.min(0))
        x2, y2 = tuple(lms.max(0))
        radius_x = 1.2 * (x2 - x1) / 2.0
        radius_y = 1.2 * (y2 - y1) / 2.0
        radius = np.array((radius_x, radius_y))
        center = (np.array((x1, y1)) + np.array((x2, y2))) / 2.0
        w, h, _ = frame.shape
        x1, y1 = clamp_to_im(center - radius, h, w)
        x2, y2 = clamp_to_im(center + radius + 1, h, w)
        offset = np.array((x1, y1))
        lms = (lms[:, 0:2] - offset).astype(np.int)
        frame = frame[y1:y2, x1:x2]
        return frame, lms, offset

    def get_eye_state(self, frame, lms):
        if self.no_gaze:
            return [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        lms = np.array(lms)
        e_x = [0,0]
        e_y = [0,0]
        scale = [0,0]
        reference = [None, None]
        angles = [0, 0]
        face_frame, lms, offset = self.extract_face(frame, lms)
        (right_eye, e_x[0], e_y[0], reference[0], angles[0]) = self.prepare_eye(face_frame, frame, False)
        (left_eye, e_x[1], e_y[1], reference[1], angles[1]) = self.prepare_eye(face_frame, frame, True)
        if right_eye is None or left_eye is None:
            return [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        both_eyes = np.concatenate((right_eye, left_eye))
        results = None
        results = self.gaze_model.run([], {self.input_name: both_eyes})
        open = [0, 0]
        open[0] = 1#results[1][0].argmax()
        open[1] = 1#results[1][1].argmax()
        results = np.array(results[0])

        eye_state = []
        for i in range(2):
            m = int(results[i][0].argmax())
            x = m // 8
            y = m % 8
            conf = float(results[i][0][x,y])

            off_x = 32.0 * logit(results[i][1][x, y], 8.0)
            off_y = 32.0 * logit(results[i][2][x, y], 8.0)
            if i == 1:
                eye_x = 32.0 * float(x) / 8.0 + off_x
            else:
                eye_x = 32.0 * float(x) / 8.0 + off_x
            eye_y = 32.0 * float(y) / 8.0 + off_y

            if self.debug_gaze:
                if i == 0:
                    frame[int(eye_y), int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), int(eye_x+1)] = (0, 0, 255)
                    frame[int(eye_y), int(eye_x+1)] = (0, 0, 255)
                else:
                    frame[int(eye_y), 32+int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), 32+int(eye_x)] = (0, 0, 255)
                    frame[int(eye_y+1), 32+int(eye_x+1)] = (0, 0, 255)
                    frame[int(eye_y), 32+int(eye_x+1)] = (0, 0, 255)

            if i == 0:
                eye_x = e_x[i] + scale[i][0] * eye_x
            else:
                eye_x = e_x[i] + scale[i][0] * (32. - eye_x)
            eye_y = e_y[i] + scale[i][1] * eye_y
            eye_x, eye_y = rotate(reference[i], (eye_x, eye_y), -angles[i])

            eye_x = eye_x + offset[0]
            eye_y = eye_y + offset[1]
            eye_state.append([open[i], eye_y, eye_x, conf])

        eye_state = np.array(eye_state)
        eye_state[np.isnan(eye_state).any(axis=1)] = np.array([1.,0.,0.,0.], dtype=np.float32)
        return eye_state

    def assign_face_info(self, results):
        if self.max_faces == 1 and len(results) == 1:
            conf, (lms, eye_state), conf_adjust = results[0]
            self.face_info[0].update((conf - conf_adjust, (lms, eye_state)), np.array(lms)[:, 0:2].mean(0), self.frame_count)
            return
        result_coords = []
        adjusted_results = []
        for conf, (lms, eye_state), conf_adjust in results:
            adjusted_results.append((conf - conf_adjust, (lms, eye_state)))
            result_coords.append(np.array(lms)[:, 0:2].mean(0))
        results = adjusted_results
        candidates = [[]] * self.max_faces
        max_dist = 2 * np.linalg.norm(np.array([self.width, self.height]))
        for i, face_info in enumerate(self.face_info):
            for j, coord in enumerate(result_coords):
                if face_info.coord is None:
                    candidates[i].append((max_dist, i, j))
                else:
                    candidates[i].append((np.linalg.norm(face_info.coord - coord), i, j))
        for i, candidate in enumerate(candidates):
            candidates[i] = sorted(candidate)
        found = 0
        target = len(results)
        used_results = {}
        used_faces = {}
        while found < target:
            min_list = min(candidates)
            candidate = min_list.pop(0)
            face_idx = candidate[1]
            result_idx = candidate[2]
            if not result_idx in used_results and not face_idx in used_faces:
                self.face_info[face_idx].update(results[result_idx], result_coords[result_idx], self.frame_count)
                min_list.clear()
                used_results[result_idx] = True
                used_faces[face_idx] = True
                found += 1
            if len(min_list) == 0:
                min_list.append((2 * max_dist, face_idx, result_idx))
        for face_info in self.face_info:
            if face_info.frame_count != self.frame_count:
                face_info.update(None, None, self.frame_count)

    def predict(self, frame, additional_faces=[]):
        self.frame_count += 1
        start = time.perf_counter()
        im = frame

        duration_fd = 0.0
        duration_pp = 0.0
        duration_model = 0.0
        duration_pnp = 0.0

        new_faces = []
        new_faces.extend(self.faces)
        bonus_cutoff = len(self.faces)
        new_faces.extend(additional_faces)
        self.wait_count += 1
        if self.detected == 0:
            start_fd = time.perf_counter()
            if self.use_retinaface > 0 or self.try_hard:
                retinaface_detections = self.retinaface.detect_retina(frame)
                new_faces.extend(retinaface_detections)
            if self.use_retinaface == 0 or self.try_hard:
                new_faces.extend(self.detect_faces(frame))
            if self.try_hard:
                new_faces.extend([(0, 0, self.width, self.height)])
            duration_fd = 1000 * (time.perf_counter() - start_fd)
            self.wait_count = 0
        elif self.detected < self.max_faces:
            if self.use_retinaface > 0:
                new_faces.extend(self.retinaface_scan.get_results())
            if self.wait_count >= self.scan_every:
                if self.use_retinaface > 0:
                    self.retinaface_scan.background_detect(frame)
                else:
                    start_fd = time.perf_counter()
                    new_faces.extend(self.detect_faces(frame))
                    duration_fd = 1000 * (time.perf_counter() - start_fd)
                    self.wait_count = 0
        else:
            self.wait_count = 0

        if len(new_faces) < 1:
            duration = (time.perf_counter() - start) * 1000
            #if not self.silent:
            #    print(f"Took {duration:.2f}ms")
            return []

        crops = []
        crop_info = []
        num_crops = 0
        for j, (x,y,w,h) in enumerate(new_faces):
            crop_x1 = x - int(w * 0.1)
            crop_y1 = y - int(h * 0.125)
            crop_x2 = x + w + int(w * 0.1)
            crop_y2 = y + h + int(h * 0.125)

            crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), self.width, self.height)
            crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), self.width, self.height)

            scale_x = float(crop_x2 - crop_x1) / self.res
            scale_y = float(crop_y2 - crop_y1) / self.res

            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                continue

            start_pp = time.perf_counter()
            crop = self.preprocess(im, (crop_x1, crop_y1, crop_x2, crop_y2))
            duration_pp += 1000 * (time.perf_counter() - start_pp)
            crops.append(crop)
            crop_info.append((crop_x1, crop_y1, scale_x, scale_y, 0.0 if j >= bonus_cutoff else 0.1))
            num_crops += 1

        #start_model = time.perf_counter()
        outputs = {}
        if num_crops == 1:
            output = self.session.run([], {self.input_name: crops[0]})[0]
            conf, lms = self.landmarks(output[0], crop_info[0])
            if conf > self.threshold:
                try:
                    eye_state = self.get_eye_state(frame, lms)
                except:
                    eye_state = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
                outputs[crop_info[0]] = (conf, (lms, eye_state), 0)
        else:
            started = 0
            results = queue.Queue()
            for i in range(min(num_crops, self.max_workers)):
                thread = threading.Thread(target=worker_thread, args=(self.sessions[started], frame, crops[started], crop_info[started], results, self.input_name, started, self))
                started += 1
                thread.start()
            returned = 0
            while returned < num_crops:
                result = results.get(True)
                if len(result) != 1:
                    session, conf, lms, sample_crop_info, idx = result
                    outputs[sample_crop_info] = (conf, lms, idx)
                else:
                    session = result[0]
                returned += 1
                if started < num_crops:
                    thread = threading.Thread(target=worker_thread, args=(session, frame, crops[started], crop_info[started], results, self.input_name, started, self))
                    started += 1
                    thread.start()

        actual_faces = []
        good_crops = []
        for crop in crop_info:
            if crop not in outputs:
                continue
            conf, lms, i = outputs[crop]
            x1, y1, _ = lms[0].min(0)
            x2, y2, _ = lms[0].max(0)
            bb = (x1, y1, x2 - x1, y2 - y1)
            outputs[crop] = (conf, lms, i, bb)
            actual_faces.append(bb)
            good_crops.append(crop)
        groups = group_rects(actual_faces)

        best_results = {}
        for crop in good_crops:
            conf, lms, i, bb = outputs[crop]
            if conf < self.threshold:
                continue;
            group_id = groups[str(bb)][0]
            if not group_id in best_results:
                best_results[group_id] = [-1, [], 0]
            if conf > self.threshold and best_results[group_id][0] < conf + crop[4]:
                best_results[group_id][0] = conf + crop[4]
                best_results[group_id][1] = lms
                best_results[group_id][2] = crop[4]

        sorted_results = sorted(best_results.values(), key=lambda x: x[0], reverse=True)[:self.max_faces]
        self.assign_face_info(sorted_results)
        #duration_model = 1000 * (time.perf_counter() - start_model)

        results = []
        detected = []
        start_pnp = time.perf_counter()
        for face_info in self.face_info:
            if face_info.alive and face_info.conf > self.threshold:
                lms = face_info.lms[:, 0:2]
                x1, y1 = tuple(lms[0:66].min(0))
                x2, y2 = tuple(lms[0:66].max(0))
                bbox = (y1, x1, y2 - y1, x2 - x1)
                face_info.bbox = bbox
                detected.append(bbox)
                results.append(face_info)
        duration_pnp += 1000 * (time.perf_counter() - start_pnp)

        if len(detected) > 0:
            self.detected = len(detected)
            self.faces = detected
            self.discard = 0
        else:
            self.detected = 0
            self.discard += 1
            if self.discard > self.discard_after:
                self.faces = []
            else:
                if self.bbox_growth > 0:
                    faces = []
                    for (x,y,w,h) in self.faces:
                        x -= w * self.bbox_growth
                        y -= h * self.bbox_growth
                        w += 2 * w * self.bbox_growth
                        h += 2 * h * self.bbox_growth
                        faces.append((x,y,w,h))
                    self.faces = faces
        self.faces = [x for x in self.faces if not np.isnan(np.array(x)).any()]
        self.detected = len(self.faces)

        #duration = (time.perf_counter() - start) * 1000
        #if not self.silent:
        #    print(f"Took {duration:.2f}ms (detect: {duration_fd:.2f}ms, crop: {duration_pp:.2f}ms, track: {duration_model:.2f}ms")

        results = sorted(results, key=lambda x: x.id)

        return results
