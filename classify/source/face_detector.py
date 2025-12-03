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


class FaceDetector(object):
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

    def pre_process(self, img, size, max_stride=32, padding=True):
        img = img_utils.decode_img(img)
        imgh, imgw = img.shape[:2]
        scale_w = 1.0
        scale_h = 1.0
        resize_w = imgw
        resize_h = imgh
        if imgh > imgw:
            resize_h = size
            resize_w = imgw * resize_h / imgh
            resize_w = int(math.ceil(resize_w / max_stride) * max_stride)
            scale_h = resize_h / imgh
            scale_w = resize_w / imgw
        else:
            resize_w = size
            resize_h = imgh * resize_w / imgw
            resize_h = int(math.ceil(resize_h / max_stride) * max_stride)
            scale_h = resize_h / imgh
            scale_w = resize_w / imgw
        if resize_w != imgw and resize_h != imgh:
            img = cv2.resize(img, dsize=(resize_w, resize_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:,:,np.newaxis]
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = img.transpose(2,0,1)
        if padding:
            batch_imgs = np.zeros(shape=(1,1,size,size), dtype=img.dtype)
            batch_imgs[0,:,:resize_h,:resize_w] = img
            begin_w = 0
            begin_h = 0
        else:
            batch_imgs = img[np.newaxis]
            begin_w = 0
            begin_h = 0
        return batch_imgs, scale_w, scale_h, begin_w, begin_h
    
    def decode(self, hm_pred, reg_pred, stride, conf_thr=0.1):
        ind = np.nonzero(hm_pred >= conf_thr)
        ind_b, ind_c, ind_y, ind_x = ind
        dfl = reg_pred[ind_b, :, ind_y, ind_x]
        dfl = dfl.reshape(-1, 4, 8)
        dfl_exp = np.exp(dfl)
        dfl_softmax = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)
        points = np.array([0., 1., 2., 3., 4., 5., 6., 7.], dtype=np.float32)
        box_dis = np.sum(dfl_softmax * points, axis=2)
        boxes = np.stack([ind_x+0.5-box_dis[:, 0], ind_y+0.5-box_dis[:,1], ind_x+0.5+box_dis[:,2], ind_y+0.5+box_dis[:,3]], axis=1)
        boxes = boxes * stride
        score = hm_pred[ind_b, ind_c, ind_y, ind_x][:, None]
        boxes = np.concatenate([boxes, score], 1)
        return boxes

    def decode_batch(self, outs,  batch_begin_wh, batch_scale_wh, conf_thr=0.1):
        strides = [4, 8, 16, 32]
        batch_size = outs[0].shape[0]
        res = []
        for batch_idx in range(batch_size):
            begin_w, begin_h = batch_begin_wh[batch_idx]
            scale_w, scale_h = batch_scale_wh[batch_idx]
            img_outs = [out[batch_idx][np.newaxis] for out in outs]
            boxes = []
            for idx, stride in enumerate(strides):
                hm_pred = img_outs[idx*2]
                reg_pred = img_outs[idx*2+1]
                stride_boxes = self.decode(hm_pred, reg_pred, stride, conf_thr=conf_thr)
                boxes.append(stride_boxes)
            boxes = np.concatenate(boxes, axis=0)

            inds = box_utils.nms(boxes, 0.3)
            boxes = boxes[inds].copy()
            boxes = boxes.reshape(-1, 5)
            boxes[:, :4] = (boxes[:, :4] - np.array([begin_w, begin_h, begin_w, begin_h], dtype=np.float32)) / np.array([scale_w, scale_h, scale_w, scale_h], dtype=np.float32)

            res.append(boxes)
        return res

    def forward(self, batch_imgs):
        input_feed = {self.input_nodes[0].name : batch_imgs}
        output_names = [node.name for node in self.output_nodes]
        outputs = self.session.run(output_names, input_feed)
        return outputs

    def pred_img(self, img, size=480, conf_thr=0.3):
        img_data, scale_w, scale_h, begin_w, begin_h = self.pre_process(img=img, size=size, padding=True)
        outs = self.forward(img_data)
        strides = [4, 8, 16, 32]
        boxes = []
        for idx, stride in enumerate(strides):
            hm_pred = outs[idx*2]
            reg_pred = outs[idx*2+1]
            stride_boxes = self.decode(hm_pred, reg_pred, stride=stride, conf_thr=conf_thr)
            boxes.append(stride_boxes)
        boxes = np.concatenate(boxes, axis=0)
        inds = box_utils.nms(boxes, 0.3)
        boxes = boxes[inds].copy()
        boxes[:, :4] /= np.array([scale_w, scale_h, scale_w, scale_h], dtype=np.float32)
        return boxes
    