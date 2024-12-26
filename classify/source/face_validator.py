import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import math
import onnxruntime
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from source.utils import utils, box_utils, img_utils


class FaceValidator(object):
    def __init__(self, onnx_path, device='cpu') -> None:
        self.onnx_path = onnx_path
        self.device = device
        self.build_model(self.onnx_path)

    def build_model(self, onnx_path):
        if self.device == 'cpu':
            providers =  ['CPUExecutionProvider']
        else:
            if self.device == 'gpu':
                cuda_provider = ('CUDAExecutionProvider', {'device_id':0})
            elif self.device.startswith('gpu:'):
                gpu_id = int(self.device.split('gpu:')[-1])
                cuda_provider = ('CUDAExecutionProvider', {'device_id':gpu_id})
            else:
                raise ValueError
            providers =  [cuda_provider, 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers,)
        self.input_nodes = self.session.get_inputs()
        self.output_nodes = self.session.get_outputs()
    
    def precess_img(self, img, size=112, keep_ratio=False, padding_val=None, mean=127.5, var=127.5):
        img = img_utils.decode_img(img)
        img_h, img_w = img.shape[:2]
        scale_w, scale_h = 1.0, 1.0
        resize_w, resize_h = img_w, img_h
        if not keep_ratio:
            resize_w = size
            resize_h = size
            scale_w = resize_w / img_w
            scale_h = resize_h / img_h
        else:
            if img_h > img_w:
                resize_h = size
                scale_h = resize_h / img_h
                resize_w = int(scale_h * img_w)
                scale_w = resize_w / img_w
            else:
                resize_w = size
                scale_w = resize_w / img_w
                resize_h = int(scale_w * img_h)
                scale_h = resize_h / img_h
        img = cv2.resize(img, dsize=(resize_w, resize_h))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=2)
        img_gray = (img_gray.astype(np.float32) - mean) / var
        img_gray = img_gray.transpose(2,0,1)#ns=(1,h,w)
        if keep_ratio or padding_val is not None:
            begin_x = (size - resize_w) // 2
            begin_y = (size - resize_h) // 2
            img_data = np.full(shape=(1, 1, size, size), fill_value=padding_val, dtype=np.float32)
            img_data[0,:,begin_y:begin_y+resize_h, begin_x:begin_x+resize_w] = img_gray
        else:
            img_data = img_gray[np.newaxis]
            begin_x = 0
            begin_y = 0
        return img_data, begin_x, begin_y, resize_w, resize_h, scale_w, scale_h

    def forward(self, batch_imgs):
        input_feed = {self.input_nodes[0].name : batch_imgs}
        output_names = [node.name for node in self.output_nodes]
        outputs = self.session.run(output_names, input_feed)
        return outputs
    
    def decode(self, outs):
        exp_outs = np.exp(outs - np.max(outs))
        probabilities = exp_outs / np.sum(exp_outs)
        return probabilities

    def pred_img(self, img, size=112, conf_thr=0.3):
        """
        单张照片推理。由于需要和训练用图保持分布一致，不能直接传图使用。请使用with box的函数。
        """
        img_data, begin_w, begin_h, resize_w, resize_h, scale_w, scale_h = self.precess_img(img=img)
        outs = self.forward(img_data)
        probabilities = self.decode(outs[0].squeeze())
        return probabilities.tolist()
    
    def pred_img_with_boxes(self, img, boxes):
        img = img_utils.decode_img(img)
        img_h, img_w = img.shape[:2]
        face_infos = []
        for box in boxes:
            e_box = box_utils.expand_box1(box[:4], img_w, img_h, expand_ratio=1.3)
            e_x1, e_y1, e_x3, e_y3 = e_box[:4]
            crop_img = img[e_y1:e_y3, e_x1:e_x3]
            probabilities = self.pred_img(img=crop_img)
            face_info = dict(
                box=box,
                validate_info=probabilities
            )
            face_infos.append(face_info)
        return face_infos
