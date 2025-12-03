import os
import os.path as osp
from PIL import Image
import numpy as np
import cv2
import json

def convert_to_buildin_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_buildin_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_buildin_types(value) for key, value in obj.items()}
    else:
        return obj
    
def save_json_api_results(save_path, results):
    json_compatible_results = convert_to_buildin_types(results)
    with open(save_path, 'w') as f:
        json.dump(json_compatible_results, f, indent=4)

def save_txt_box_res(img_id, txt_path, boxes):
    makedirs(osp.dirname(txt_path))
    with open(txt_path, 'w') as f:
        f.write(f'{img_id}\n')
        f.write('{}\n'.format(len(boxes)))
        for box in boxes:
            line = '{:.2f} {:.2f} {:.2f} {:.2f} {:.3f}'.format(box[0], box[1], box[2], box[3], box[4])
            f.write(line+'\n')


def save_json_faceinfo_res(json_path, face_infos):
    makedirs(osp.dirname(json_path))
    for face_info in face_infos:
        face_info["box"] = face_info["box"].tolist()
        if "lmk_info" in face_info:
            for key in face_info["lmk_info"]:
                face_info["lmk_info"][key] = face_info["lmk_info"][key].tolist() if \
                        isinstance(face_info["lmk_info"][key], np.ndarray) else float(face_info["lmk_info"][key])

    json_file = open(json_path, mode='w')
    json.dump(face_infos, json_file, ensure_ascii=False, indent=4)

def write_json(json_path, record:dict):
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=4)

def write_cn_json(json_path, record:dict):
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=4, ensure_ascii=False)


def makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_file_folder(file_path):
    dir_path = osp.dirname(file_path)
    makedirs(dir_path=dir_path)

def get_img_paths_from_dir(img_dir, ext='.jpg'):
    img_paths = []
    for parent, dirnames, filenames in os.walk(img_dir):
        for filename in filenames:
            if filename.endswith(ext):
                img_path = osp.join(parent, filename)
                img_paths.append(img_path)
    return img_paths

def mask_dilation(mask:Image.Image) -> Image.Image:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    mask = cv2.dilate(mask, kernel)
    return mask

def mask_min_rect_rectangle(mask):
    mask = np.asarray(mask)
    index = np.nonzero(mask)
    if len(index[0])!=0 and len(index[1])!=0:
        minx = np.min(index[1])
        maxx = np.max(index[1])
        miny = np.min(index[0])
        maxy = np.max(index[0])
    return [minx, miny, maxx, maxy]

def show_mask(mask: np.ndarray,
              image: np.ndarray,
              random_color: bool = False) -> np.ndarray:
    """Visualize a mask on top of an image.

    Args:
        mask (np.ndarray): A 2D array of shape (H, W).
        image (np.ndarray): A 3D array of shape (H, W, 3).
        random_color (bool): Whether to use a random color for the mask.
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the mask
        visualized on top of the image.
    """
    if random_color:
        color = np.concatenate([np.random.random(3)], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

    image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), 0.3, 0)
    return image


def show_points(coords: np.ndarray, labels: np.ndarray,
                image: np.ndarray) -> np.ndarray:
    """Visualize points on top of an image.

    Args:
        coords (np.ndarray): A 2D array of shape (N, 2).
        labels (np.ndarray): A 1D array of shape (N,).
        image (np.ndarray): A 3D array of shape (H, W, 3).
    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with the points
        visualized on top of the image.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for p in pos_points:
        image = cv2.circle(
            image, p.astype(int), radius=5, color=(0, 255, 0), thickness=-1)
    for p in neg_points:
        image = cv2.circle(
            image, p.astype(int), radius=5, color=(255, 0, 0), thickness=-1)
    return image

def read_img(img_path, mode='BGR'):
    if mode == 'BGR':
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    elif mode == 'GRAY':
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    elif mode == 'RGB':
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        
    return img

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

        