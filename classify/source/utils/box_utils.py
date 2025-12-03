import os
import os.path as osp
import math
import numpy as np
from . import utils

def nms(dets, thresh=0.5):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def expand_box1(box:list, img_w=None, img_h=None, expand_ratio=1.3):
    x1, y1, x3, y3 = box[:4]
    box_cx, box_cy = (x1 + x3) / 2, (y1 + y3) / 2
    box_w, box_h = x3 - x1, y3 - y1
    max_wh = max(box_w, box_h)
    e_wh = max_wh * expand_ratio
    e_x1 = int(box_cx - e_wh / 2)
    e_y1 = int(box_cy - e_wh / 2)
    e_x3 = int(box_cx + e_wh / 2)
    e_y3 = int(box_cy + e_wh / 2)

    if img_w is not None or img_h is not None:
        e_x1 = max(0, e_x1)
        e_y1 = max(0, e_y1)
        e_x3 = min(e_x3, img_w)
        e_y3 = min(e_y3, img_h)
    return [e_x1, e_y1, e_x3, e_y3]

def expand_box2(box:list, img_w=None, img_h=None, expand_ratio=1.3, use_side="max"):
    x1, y1, x3, y3 = box[:4]
    box_cx, box_cy = (x1 + x3) / 2, (y1 + y3) / 2
    box_w, box_h = x3 - x1, y3 - y1
    if use_side == "max":
        side = max(box_w, box_h)
    elif use_side == "min":
        side = min(box_w, box_h)
    elif use_side == "avg":
        side = (box_w + box_h) / 2
    else:
        raise ValueError
    e_wh = side * expand_ratio
    e_x1 = int(box_cx - e_wh / 2)
    e_y1 = int(box_cy - e_wh / 2)
    e_x3 = int(box_cx + e_wh / 2)
    e_y3 = int(box_cy + e_wh / 2)

    if img_w is not None or img_h is not None:
        e_x1 = max(0, e_x1)
        e_y1 = max(0, e_y1)
        e_x3 = min(e_x3, img_w-1)
        e_y3 = min(e_y3, img_h-1)
    return [e_x1, e_y1, e_x3, e_y3]

def expand_box3(box:list, img_w=None, img_h=None, expand_ratio=1.3):
    x1, y1, x3, y3 = box[:4]
    box_cx, box_cy = (x1 + x3) / 2, (y1 + y3) / 2
    box_w, box_h = x3 - x1, y3 - y1
    target_w = box_w*expand_ratio
    target_h = box_h*expand_ratio
    # max_wh = max(box_w, box_h)
    # e_wh = max_wh * expand_ratio
    e_x1 = int(box_cx - target_w / 2)
    e_y1 = int(box_cy - target_h / 2)
    e_x3 = int(box_cx + target_w / 2)
    e_y3 = int(box_cy + target_h / 2)
 
    if img_w is not None or img_h is not None:
        e_x1 = max(0, e_x1)
        e_y1 = max(0, e_y1)
        e_x3 = min(e_x3, img_w)
        e_y3 = min(e_y3, img_h)
    return [e_x1, e_y1, e_x3, e_y3]

def expand_box4(box:list, img_w=None, img_h=None, expand_w_ratio=1.0, expand_h_ratio=1.0):
    x1, y1, x3, y3 = box[:4]
    box_cx, box_cy = (x1 + x3) / 2, (y1 + y3) / 2
    box_w, box_h = x3 - x1, y3 - y1
    target_w = box_w * expand_w_ratio
    target_h = box_h * expand_h_ratio
    # max_wh = max(box_w, box_h)
    # e_wh = max_wh * expand_ratio
    e_x1 = int(box_cx - target_w / 2)
    e_y1 = int(box_cy - target_h / 2)
    e_x3 = int(box_cx + target_w / 2)
    e_y3 = int(box_cy + target_h / 2)
 
    if img_w is not None or img_h is not None:
        e_x1 = max(0, e_x1)
        e_y1 = max(0, e_y1)
        e_x3 = min(e_x3, img_w)
        e_y3 = min(e_y3, img_h)
    return [e_x1, e_y1, e_x3, e_y3]

def get_box_infos(box):
    x1,y1,x3,y3 = box[:4]
    w = x3 - x1
    h = y3 - y1
    area = w * h
    return w, h, area

def get_points_distance(point1, point2):
    dis = math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)    
    dis = math.sqrt(dis)
    return dis

def get_symmetry_point(point1, center_point):
    cx = center_point[0]*2 - point1[0]
    cy = center_point[1]*2 - point1[1]
    return [cx, cy]

def get_points_center(point1, point2):
    cx = (point1[0] + point2[0]) / 2
    cy = (point1[1] + point2[1]) / 2
    if isinstance(point1, list):
        return [cx, cy]
    elif isinstance(point1, np.ndarray):
        return np.array([cx, cy], dtype=np.float32)
    else:
        return [cx, cy]
    
def get_point_at_line(k, b, x=None, y=None):
    if x is None:
        res = (y - b) / k
        return [res, y]
    elif y is None:
        res = k * x + b
        return [x, res]
    else:
        raise ValueError
    
def get_line_param(p1, p2, x=None, y=None):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - k * p2[0]
    if x is not None:
        res = k * x + b
        return res
    elif y is not None:
        res = (y - b) / k 
        return res
    else:
        return k, b
    
def get_lines_intersection_point(k1, b1, k2, b2):
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return [x,y]

def get_face_info_in_crop_img(face_info:dict, crop_box:list):
    crop_x1,crop_y1,crop_x3,crop_y3 = crop_box[:4]
    if face_info.get("box", None) is not None:
        box = face_info["box"]
        box[0] -= crop_x1
        box[1] -= crop_y1
        box[2] -= crop_x1
        box[3] -= crop_y1
        face_info["box"] = box
    if face_info.get("lmk_info", None) is not None:
        lmk_info = face_info["lmk_info"]
        lmk_xy = lmk_info["lmk_xy"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] -= crop_x1
            lmk_xy[lmk_idx][1] -= crop_y1
        lmk_info["lmk_xy"] = lmk_xy
        face_info["lmk_info"] = lmk_info
    return face_info

def scale_face_info(face_info:dict, scale_w, scale_h):
    if face_info.get("box", None) is not None:
        box = face_info["box"]
        box[0] *= scale_w
        box[1] *= scale_h
        box[2] *= scale_w
        box[3] *= scale_h
        face_info["box"] = box
    if face_info.get("lmk_info", None) is not None:
        lmk_info = face_info["lmk_info"] 
        lmk_xy = lmk_info["lmk_xy"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] *= scale_w
            lmk_xy[lmk_idx][1] *= scale_h
        lmk_info["lmk_xy"] = lmk_xy
        face_info["lmk_info"] = lmk_info
    return face_info

def get_facebody_info_in_crop_img(facebody_info:dict, crop_box:list):
    crop_x1,crop_y1,crop_x3,crop_y3 = crop_box[:4]
    if facebody_info["face_info"] is not None and facebody_info["face_info"].get("box", None) is not None:
        box = facebody_info["face_info"]["box"]
        box[0] -= crop_x1
        box[1] -= crop_y1
        box[2] -= crop_x1
        box[3] -= crop_y1
        facebody_info["face_info"]["box"] = box
    if facebody_info["face_info"] is not None and facebody_info["face_info"].get("lmk_info", None) is not None:
        lmk_info = facebody_info["face_info"]["lmk_info"]
        lmk_xy = lmk_info["lmk_xy"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] -= crop_x1
            lmk_xy[lmk_idx][1] -= crop_y1
        lmk_info["lmk_xy"] = lmk_xy
        facebody_info["face_info"]["lmk_info"] = lmk_info
    if facebody_info["body_info1"].get("body_landmark", None) is not None:
        lmk_xy = facebody_info["body_info1"]["body_landmark"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] -= crop_x1
            lmk_xy[lmk_idx][1] -= crop_y1
        facebody_info["body_info1"]["body_landmark"] = lmk_xy
    if facebody_info["body_info2"].get("body_landmark", None) is not None:
        lmk_xy = facebody_info["body_info2"]["body_landmark"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] -= crop_x1
            lmk_xy[lmk_idx][1] -= crop_y1
        facebody_info["body_info2"]["body_landmark"] = lmk_xy
    if facebody_info["body_info2"].get("body_box", None) is not None:
        box = facebody_info["body_info2"]["body_box"]
        box[0] -= crop_x1
        box[1] -= crop_y1
        box[2] -= crop_x1
        box[3] -= crop_y1
        facebody_info["body_info2"]["body_box"] = box
    return facebody_info


def scale_facebody_info(facebody_info:dict, scale_w, scale_h):
    if facebody_info["face_info"] is not None and facebody_info["face_info"].get("box", None) is not None:
        box = facebody_info["face_info"]["box"]
        box[0] *= scale_w
        box[1] *= scale_h
        box[2] *= scale_w
        box[3] *= scale_h
        facebody_info["face_info"]["box"] = box
    if facebody_info["face_info"] is not None and facebody_info["face_info"].get("lmk_info", None) is not None:
        lmk_info = facebody_info["face_info"]["lmk_info"]
        lmk_xy = lmk_info["lmk_xy"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] *= scale_w
            lmk_xy[lmk_idx][1] *= scale_h
        lmk_info["lmk_xy"] = lmk_xy
        facebody_info["face_info"]["lmk_info"] = lmk_info
    if facebody_info["body_info1"].get("body_landmark", None) is not None:
        lmk_xy = facebody_info["body_info1"]["body_landmark"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] *= scale_w
            lmk_xy[lmk_idx][1] *= scale_h
        facebody_info["body_info1"]["body_landmark"] = lmk_xy
    if facebody_info["body_info2"].get("body_landmark", None) is not None:
        lmk_xy = facebody_info["body_info2"]["body_landmark"]
        for lmk_idx in range(len(lmk_xy)):
            lmk_xy[lmk_idx][0] *= scale_w
            lmk_xy[lmk_idx][1] *= scale_h
        facebody_info["body_info2"]["body_landmark"] = lmk_xy
    if facebody_info["body_info2"].get("body_box", None) is not None:
        box = facebody_info["body_info2"]["body_box"]
        box[0] *= scale_w
        box[1] *= scale_h
        box[2] *= scale_w
        box[3] *= scale_h
        facebody_info["body_info2"]["body_box"] = box
    return facebody_info


def expand_rotate_box(rotate_box:np.ndarray, face_info:dict):
    rotate_box = rotate_box.tolist()
    roate_box_x1y1 = rotate_box[0]
    roate_box_x2y2 = rotate_box[1]
    roate_box_x3y3 = rotate_box[2]
    roate_box_x4y4 = rotate_box[3]
    roate_box_x1y1_x4y4_center = get_points_center(roate_box_x1y1, roate_box_x4y4)
    roate_box_x2y2_x3y3_center = get_points_center(roate_box_x2y2, roate_box_x3y3)
    left_key_point = face_info["lmk_info"]["lmk_xy"][9]
    right_key_point = face_info["lmk_info"]["lmk_xy"][25]
    if roate_box_x1y1_x4y4_center[0] > left_key_point[0]:
        k1, b1 = get_line_param(roate_box_x1y1, roate_box_x4y4)
        k2, b2 = get_line_param(roate_box_x1y1, roate_box_x2y2)
        k3, b3 = get_line_param(roate_box_x4y4, roate_box_x3y3)
        b1 = left_key_point[1] - k1 * left_key_point[0]
        roate_box_x1y1_extend = get_lines_intersection_point(k1,b1,k2,b2)
        roate_box_x4y4_extend = get_lines_intersection_point(k1,b1,k3,b3)
        roate_box_x1y1_extend_extend = get_point_at_line(
            k=k2, 
            b=b2, 
            x=roate_box_x1y1_extend[0] - (roate_box_x2y2[0] - roate_box_x1y1_extend[0]) * 0.05, 
            y=None
        )
        roate_box_x4y4_extend_extend = get_point_at_line(
            k=k3, 
            b=b3, 
            x=roate_box_x4y4_extend[0] - (roate_box_x3y3[0] - roate_box_x4y4_extend[0]) * 0.05, 
            y=None
        )
        rotate_box.insert(0, roate_box_x1y1_extend_extend)
        rotate_box.append(roate_box_x4y4_extend_extend)
    elif roate_box_x2y2_x3y3_center[0] < right_key_point[0]:
        k1, b1 = get_line_param(roate_box_x2y2, roate_box_x3y3)
        k2, b2 = get_line_param(roate_box_x1y1, roate_box_x2y2)
        k3, b3 = get_line_param(roate_box_x4y4, roate_box_x3y3)
        b1 = right_key_point[1] - k1 * right_key_point[0]
        roate_box_x2y2_extend = get_lines_intersection_point(k1,b1,k2,b2)
        roate_box_x3y3_extend = get_lines_intersection_point(k1,b1,k3,b3)

        roate_box_x2y2_extend_extend = get_point_at_line(
            k=k2, 
            b=b2, 
            x=roate_box_x2y2_extend[0] + (roate_box_x2y2_extend[0] - roate_box_x1y1[0]) * 0.05, 
            y=None
        )
        roate_box_x3y3_extend_extend = get_point_at_line(
            k=k3, 
            b=b3, 
            x=roate_box_x3y3_extend[0] + (roate_box_x3y3_extend[0] - roate_box_x4y4[0]) * 0.05, 
            y=None
        )

        rotate_box.insert(2, roate_box_x2y2_extend_extend)
        rotate_box.insert(3, roate_box_x3y3_extend_extend)
    else:
        pass
    rotate_box = np.array(rotate_box)
    return rotate_box

def get_facelmk_min_box(lmk_xy):
    if isinstance(lmk_xy, list):
        lmk_xy = np.array(lmk_xy).reshape(-1, 2)
    assert isinstance(lmk_xy, np.ndarray) and lmk_xy.ndim == 2
    x1 = np.min(lmk_xy[:, 0])
    y1 = np.min(lmk_xy[:, 1])
    x3 = np.max(lmk_xy[:, 0])
    y3 = np.max(lmk_xy[:, 1])
    x1,y1,x3,y3 = list(map(int, [x1,y1,x3,y3]))
    return [x1,y1,x3,y3]
