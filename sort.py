# Simple SORT tracker implementation (fixed and defensive)
# Source: https://github.com/abewley/sort
# Modified for direct use with YOLO

import numpy as np
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    """
    Always returns an (N,2) int numpy array of matches (row, col).
    If no matches, returns np.empty((0,2), dtype=int).
    Tries lap (faster), falls back to scipy; if both fail, returns empty.
    """
    try:
        import lap
        # lapjv returns assignment arrays where x[i] is assigned col for row i or -1
        _, x, _ = lap.lapjv(cost_matrix, extend_cost=True)
        matches = []
        for i in range(len(x)):
            if int(x[i]) >= 0:
                matches.append([int(i), int(x[i])])
        if len(matches) == 0:
            return np.empty((0, 2), dtype=int)
        return np.array(matches, dtype=int)
    except Exception:
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if len(row_ind) == 0:
                return np.empty((0, 2), dtype=int)
            return np.vstack((row_ind, col_ind)).T.astype(int)
        except Exception:
            # If something unexpected happens, return empty matches (robust fallback)
            return np.empty((0, 2), dtype=int)

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    denom = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
             (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return wh / denom

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        out = np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1,4))
        if score is None:
            return out
        else:
            return np.append(out, score).reshape((1,5))

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:4] = pos
            trk[4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                inds = np.where(matched[:, 1] == t)[0]
                if len(inds) > 0:
                    d = matched[inds[0], 0]
                    trk.update(dets[d, :4])

        # create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # prepare return: only confirmed tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    """
    Returns:
        matches: (M,2) array of matched pairs (det_index, trk_index)
        unmatched_dets: 1-D array of detection indices not matched
        unmatched_trks: 1-D array of tracker indices not matched
    """
    if len(trks) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(dets)), np.empty((0), dtype=int)

    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)

    # run Hungarian / linear assignment on negative IoU (maximizes IoU)
    matched_indices = linear_assignment(-iou_matrix)

    # Defensive handling: ensure (N,2) shape
    matched_indices = np.asarray(matched_indices, dtype=int)
    if matched_indices.ndim == 1:
        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        elif matched_indices.size == 2:
            matched_indices = matched_indices.reshape(1, 2)
        elif matched_indices.size % 2 == 0:
            matched_indices = matched_indices.reshape(-1, 2)
        else:
            # fallback to empty if shape is unexpected
            matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]] if matched_indices.size else list(range(len(dets)))
    unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]] if matched_indices.size else list(range(len(trks)))

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(int(m[0]))
            unmatched_trks.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0).astype(int)

    return matches, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)
