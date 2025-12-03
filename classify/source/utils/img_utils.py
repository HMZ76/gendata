import os
import os.path as osp
from typing import List, Optional, Union
import cv2
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import base64
import math
import warnings

class_colors = [
    [255,10,30],
    [190, 190, 190],
    [237, 149, 100],
    [128, 0, 0],
    [255, 112, 132],
    [0, 0, 255],
    [255, 255, 0],
    [0, 100, 0],
    [127, 255, 0],
    [0, 255, 255],
    [92, 92, 205],
    [0, 165, 255],
    [255, 0, 255],
    [240, 32, 160],
    [173, 222, 255],
    [147, 20, 255],
    [0, 205, 205],
    [0, 139, 0],
    [102, 205, 0],
    [212, 255, 127],
    [139, 134, 0],
]


def read_img(img_path, output_type='cv2', mode='BGR'):
    if output_type == 'cv2':
        img = cv2.imread(img_path)
        if mode == 'BGR':
            img = img
        elif mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError
    elif output_type == 'pil':
        img = Image.open(img_path)
        if mode == 'RGB':
            img = img
        elif mode == 'BGR':
            img = img.convert('BGR')
        elif mode == 'GRAY':
            img = img.convert('L')
        else:
            raise NotImplementedError
    return img

def decode_img(img):
    if isinstance(img, str):
        if not osp.exists(img):
            raise FileExistsError(f'{img}')
        img = read_img(img_path=img)
    elif isinstance(img, np.ndarray):
        assert img.ndim == 3 and img.shape[2] == 3
    elif isinstance(img, Image.Image):
        if img.mode != "RGB":
            raise ValueError
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError
    return img

def visualize_face_box(img, boxes):
    img = decode_img(img)
    for box in boxes:
        x1,y1,x3,y3 = list(map(int, box[:4]))
        score = box[4]
        cv2.rectangle(img, (x1,y1), (x3,y3), color=(0,0,255), thickness=2)
        cv2.putText(img, '{:.2f}'.format(score), (x1,y1-10),  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
    return img

def visualize_face_head_box(img, face_boxes=None, head_boxes=None):
    img = decode_img(img)
    if face_boxes is not None:
        for box in face_boxes:
            x1,y1,x3,y3 = list(map(int, box[:4]))
            score = box[4]
            cv2.rectangle(img, (x1,y1), (x3,y3), color=(0,0,255), thickness=2)
            cv2.putText(img, 'face {:.2f}'.format(score), (x1,y1-10),  fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(160,150,254), thickness=2)
    if head_boxes is not None:
        for box in head_boxes:
            x1,y1,x3,y3 = list(map(int, box[:4]))
            score = box[4]
            cv2.rectangle(img, (x1,y1), (x3,y3), color=(0,230,50), thickness=2)
            cv2.putText(img, 'head {:.2f}'.format(score), (x1,y1-10),  fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(99,186,69), thickness=2)
    return img

def visualize_face_box_landmark(img, face_infos, vis_landmark_id=False):
    img = decode_img(img)
    for face_info in face_infos:
        box = list(map(int, face_info["box"]))
        lmk_xy = face_info["lmk_info"]["lmk_xy"]
        lmk_vis_score = face_info["lmk_info"]["lmk_vis_score"]
        num_lmks = lmk_xy.shape[0]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 128), 2)
        for lmk_idx in range(num_lmks):
            x, y = list(map(int, lmk_xy[lmk_idx]))
            color = (255,0,0)
            if lmk_vis_score[lmk_idx] > 0.4:
                color = (255,0,0)  #可见 - 蓝色
            else:
                color = (0,0,255)  #不可见 - 红色
            cv2.circle(img, center=(x,y), radius=2, color=color, thickness=-1)
            if vis_landmark_id:
                cv2.putText(img, str(lmk_idx), (x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(200,50,0), thickness=1)
    return img

def visualize_face_box_valid_mark(img, face_infos):
    img = decode_img(img)
    for face_info in face_infos:
        x1,y1,x3,y3,_ = list(map(int, face_info["box"]))
        score = face_info["box"][4]
        valid_score = face_info["validate_info"][0]
        cv2.rectangle(img, (x1, y1), (x3, y3), (0,0,255), 5)
        cv2.putText(img, '{:.2f}'.format(score), (x1,y1-10),  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
        cv2.putText(img, '{:.2f}'.format(valid_score), (x1,y3+25),  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(140,255,30), thickness=2)
    return img

def visualize_api_results(img, results):
    img = decode_img(img)
    if results["face_boxes"] is not None:
        img = visualize_face_head_box(img=img, face_boxes=results["face_boxes"], head_boxes=results["head_boxes"])
    if results["face_validation_info"] is not None:
        face_infos = results["face_validation_info"]
        for face_info in face_infos:
            x1,y1,x3,y3,_ = list(map(int, face_info["box"]))
            valid_score = face_info["validate_info"][0]
            cv2.putText(img, 'val {:.2f}'.format(valid_score), (x1,y3+25),  fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(97,70,28), thickness=2)
    if results["face_landmark_info"] is not None:
        face_infos = results["face_landmark_info"] 
        for face_info in face_infos:
            lmk_xy = face_info["lmk_info"]["lmk_xy"]
            lmk_vis_score = face_info["lmk_info"]["lmk_vis_score"]
            num_lmks = lmk_xy.shape[0]
            color = (148,137,69)
            for lmk_idx in range(num_lmks):
                x, y = list(map(int, lmk_xy[lmk_idx]))
                if lmk_vis_score[lmk_idx] > 0.4:
                    color = (174,221,129)  #可见 - 蓝色
                else:
                    color = (18,87,220)  #不可见 - 红色
                cv2.circle(img, center=(x,y), radius=2, color=color, thickness=-1)
    if results["face_age_gender_info"] is not None:
        face_infos = results["face_age_gender_info"]
        for face_info in face_infos:
            x1,y1,x3,y3,_ = list(map(int, face_info["box"]))
            age = face_info["age_info"]
            gender = face_info["gender_info"]
            cv2.putText(img, 'age {:.2f}'.format(age), (x1,y3+60),  fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(254,67,101), thickness=2)
            cv2.putText(img, 'gen {:.1f},{:.1f}'.format(gender[0],gender[1]), (x1,y3+95),  fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(63,118,253), thickness=2)

    return img

def img_to_HWC3(img:Union[np.ndarray, None]):
    assert img.dtype == np.uint8
    if img.ndim == 2:
        img = img[:,:,None]#ns=(h,w,1)
    assert img.ndim == 3
    h, w, c = img.shape
    assert c == 1 or c == 3 or c == 4
    if c == 3:
        return img
    if c == 1:
        img = np.concatenate([img,img,img], axis=2)
        return img
    if c == 4:
        color = img[:,:,0:3].astype(np.float32)
        alpha = img[:,:,3:4].astype(np.float32) / 255.0
        new_img = color * alpha + 255.0 * (1.0 - alpha)
        new_img = new_img.clip(0, 255).astype(np.uint8)
        return new_img

def resize_img(
        img, 
        size:Union[int, List[int]]=512, 
        keep_ratio=True, 
        keep_max_side_resize=True,
        stride:Optional[int]=None
    ):
        if isinstance(size, list) and keep_ratio == True:
            warnings.warn('size = [] is conflict with keep_ratio = {}, now keep_ratio is invalid'.format(size, keep_ratio))
            keep_ratio = False

        img_h, img_w = img.shape[:2]
        scale_w = 1.0
        scale_h = 1.0
        resize_w, resize_h = img_w, img_h
        if not keep_ratio:
            if isinstance(size, int):
                resize_w = size
                resize_h = size
            elif isinstance(size, list):
                assert len(size) == 2
                resize_w, resize_h = size 
        else:
            assert isinstance(size, int)
            if img_w >= img_h:
                if keep_max_side_resize:
                    resize_w = size
                    scale_w = resize_w / img_w
                    resize_h = math.floor(scale_w * img_h)
                    scale_h = resize_h / img_h
                else:
                    resize_h = size
                    scale_h = resize_h / img_h
                    resize_w = math.floor(scale_h * img_w)
                    scale_w = resize_w / img_w
            else:
                #imgh > imgw
                if keep_max_side_resize:
                    #imgh -> size
                    resize_h = size
                    scale_h = resize_h / img_h
                    resize_w = math.floor(scale_h * img_w)
                    scale_w = resize_w / img_w
                else:
                    #imgw -> size
                    resize_w = size
                    scale_w = resize_w / img_w
                    resize_h = math.floor(scale_w * img_h)
                    scale_h = resize_h / img_h

        if stride is not None:
            resize_w = round(resize_w / stride) * stride
            resize_h = round(resize_h / stride) * stride
        scale_w = resize_w / img_w
        scale_h = resize_h / img_h
        img = cv2.resize(img, dsize=(resize_w, resize_h))
        return img, resize_w, resize_h, scale_w, scale_h

def limit_img_maxside(img, maxside=1024, keep_ratio=True, keep_maxside=False, resize=False):
    if isinstance(img, str):
        img = read_img(img, mode='BGR')
    elif isinstance(img, np.ndarray):
        img = img
    
    img_h, img_w = img.shape[:2]
    scale_h, scale_w = 1.0, 1.0
    resize_h, resize_w = img_h, img_w

    max_wh = max(img_w, img_h)
    if max_wh > maxside or keep_maxside:
        if img_h >= img_w:
            if keep_ratio:
                resize_h = maxside
                scale_h = resize_h / img_h
                scale_w = scale_h
                resize_w = int(scale_w * img_w)
            else:
                resize_w, resize_h = maxside, maxside
                scale_w = resize_w / img_w
                scale_h = resize_h / img_h
            
        elif img_w > img_h:
            if keep_ratio:
                resize_w = maxside
                scale_w = resize_w / img_w
                scale_h = scale_w
                resize_h = int(scale_h * img_h)
            else:
                resize_w, resize_h = maxside, maxside
                scale_w = resize_w / img_w
                scale_h = resize_h / img_h
        else:
            pass
    if resize:
        if resize_w != img_w or resize_h != img_h:
            img_resize = cv2.resize(img, dsize=(resize_w, resize_h))
        else:
            img_resize = img

        return img_resize, resize_w, resize_h, scale_w, scale_h, img_w, img_h
    else:
        return None, resize_w, resize_h, scale_w, scale_h, img_w, img_h


def debug_vis_img(img, boxes=None, mask=None, contours=None, save_path=None, return_pil=False):
    if isinstance(img, Image.Image):
        assert img.mode == "RGB"
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if mask is not None:
        img_h, img_w = img.shape[:2]
        mask_color = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        mask_color[mask==255] = np.array([0,0,255], np.uint8)
        img = img.astype(np.float32) * 0.4 + mask_color.astype(np.float32) * 0.6
        img = img.astype(np.uint8)

    if boxes is not None:
        for box_idx, box in enumerate(boxes):
            x1,y1,x3,y3 = list(map(int, box[:4]))
            cv2.rectangle(img, (x1,y1), (x3,y3), color=class_colors[box_idx], thickness=2)    

    if contours is not None:
        cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,0,255), thickness=2)

    if save_path is not None:
        cv2.imwrite(save_path, img)

    if return_pil:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode="RGB")
    return img

def debug_vis_eye(img, face_infos, return_pil=False):
    if isinstance(img, Image.Image):
        assert img.mode == "RGB"
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for face_info in face_infos:
        box = face_info["box"]
        eye_info = face_info["eye_info"]
        eye_box = eye_info["eye_box"]
        open_score = eye_info["open_score"]
        open_flag = eye_info["open_flag"]
        x1,y1,x3,y3 = list(map(int, box[:4]))
        cv2.rectangle(img, (x1,y1), (x3,y3), color=(255,0,0), thickness=2)    
        x1,y1,x3,y3 = list(map(int, eye_box[:4]))
        cv2.rectangle(img, (x1,y1), (x3,y3), color=(0,255,0), thickness=2)    
        cv2.putText(img, f"{open_score:.2f}", (x3,y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0) if open_flag else (0,0,255), thickness=2)

    if return_pil:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode="RGB")
    return img

def show_img(img, tool="cv2", mode="BGR", win_name="img", wait=0, maxside=None):
    if tool == "cv2":
        if isinstance(img, np.ndarray):
            if mode == "BGR":
                pass
            elif mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(img, Image.Image):
            if img.mode == "RGB":
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                img = np.array(img)
        else:
            raise ValueError
        if maxside is not None:
            img_resize, resize_w, resize_h, scale_w, scale_h, img_w, img_h = limit_img_maxside(img=img, maxside=maxside, keep_ratio=True, keep_maxside=False, resize=True)
        else:
            img_resize = img
        cv2.imshow(win_name, img_resize)
        if wait is not None:
            cv2.waitKey(wait)
        
def img_cv2pil(img, mode="RGB", cv2pil=True):
    if cv2pil:
        if mode == "RGB":
            img = Image.fromarray(img, mode="RGB")
        elif mode == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img, mode="RGB")
        else:
            raise ValueError
    else:
        if mode == "RGB":
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif mode == "BGR":
            img = np.array(img)
        else:
            raise ValueError
    return img

def show_face_infos(img, face_infos):
    if isinstance(img, Image.Image):
        img = img_cv2pil(img=img, mode=img.mode, cv2pil=False)
    for face_info in face_infos:
        box = face_info["box"]
        lmk_info = face_info["lmk_info"]
        lmk_xy = lmk_info["lmk_xy"]
        x1,y1,x3,y3 = list(map(int, box[:4]))
        cv2.rectangle(img, (x1,y1), (x3,y3), color=(255,0,0), thickness=2)
        for lmk_idx in range(len(lmk_xy)):
            x, y = list(map(int, lmk_xy[lmk_idx]))
            cv2.circle(img, (x,y), radius=2, color=(0,255,0), thickness=-1)
    
    cv2.imshow("show_face_infos", img)
    cv2.waitKey(0)


def convert_image_to_base64(img):
    bytes_io = BytesIO()
    img.save(bytes_io, format="JPEG")
    img_base64 = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
    return img_base64

def convert_base64_to_image(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_bytes))
    return img

def debug_vis_img(img, boxes=None, mask=None, contours=None, save_path=None, return_pil=False):
    if isinstance(img, Image.Image):
        assert img.mode == "RGB"
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if isinstance(mask, Image.Image):
        assert mask.mode == "L"
        mask = np.array(mask)

    if mask is not None:
        img_h, img_w = img.shape[:2]
        mask_color = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        mask_color[mask==255] = np.array([0,0,255], np.uint8)
        img = img.astype(np.float32) * 0.4 + mask_color.astype(np.float32) * 0.6
        img = img.astype(np.uint8)

    if boxes is not None:
        for box_idx, box in enumerate(boxes):
            x1,y1,x3,y3 = list(map(int, box[:4]))
            cv2.rectangle(img, (x1,y1), (x3,y3), color=class_colors[box_idx], thickness=2)    

    if contours is not None:
        cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,0,255), thickness=2)

    if save_path is not None:
        cv2.imwrite(save_path, img)

    if return_pil:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode="RGB")
    return img