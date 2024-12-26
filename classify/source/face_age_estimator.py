import os
import os.path as osp
import onnxruntime
import cv2
import math
from PIL import Image
import numpy as np
from source.utils import utils, box_utils, img_utils

class FaceAgeEstimator(object):
    def __init__(self, onnx_path, device='cpu') -> None:
        self.onnx_path = onnx_path
        self.device = device

        self.build_model()
        self.input_nodes = self.session.get_inputs()
        self.output_nodes = self.session.get_outputs()
        self.rgb_input = False if "gray" in self.onnx_path else True

        self.size = 112
        self.stride = 16

    def build_model(self):
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

        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=providers)
        print('load onnx model from {}!'.format(self.onnx_path))
    
    def align_img(self, img, landmarks):
        left_eye = landmarks[93]
        right_eye = landmarks[35]
        degree = math.degrees(math.atan2(left_eye[1]-right_eye[1], left_eye[0]-right_eye[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        img = img.rotate(degree, resample=Image.BICUBIC, expand=False)
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
        return img

    def precess_img(self, img, size=112, keep_ratio=False, padding_val=None, mean=127.5, var=127.5, input_rgb=False):
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
        if self.rgb_input==False:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img_gray, axis=2)
        img = (img.astype(np.float32) - mean) / var
        img = img.transpose(2,0,1)#ns=(1,h,w)
        if keep_ratio or padding_val is not None:
            begin_x = (size - resize_w) // 2
            begin_y = (size - resize_h) // 2
            if self.rgb_input==False:
                img_data = np.full(shape=(1, 1, size, size), fill_value=padding_val, dtype=np.float32)
            else:
                img_data = np.full(shape=(1, 3, size, size), fill_value=padding_val, dtype=np.float32)
            img_data[0,:,begin_y:begin_y+resize_h, begin_x:begin_x+resize_w] = img
        else:
            img_data = img[np.newaxis]
            begin_x = 0
            begin_y = 0
        return img_data, begin_x, begin_y, resize_w, resize_h, scale_w, scale_h
    
    def forward(self, batch_img):
        input_feed = {self.input_nodes[0].name : batch_img}
        output_names = [node.name for node in self.output_nodes]
        outputs = self.session.run(output_names, input_feed)
        return outputs
    
    def pred_img(self, img):
        """
        单张照片推理。由于需要和训练用图保持分布一致，不能直接传图使用。请使用with box的函数。
        """
        img_data, begin_x, begin_y, resize_w, resize_h, scale_w, scale_h = self.precess_img(img, size=112, keep_ratio=True, padding_val=0)
        outs = self.forward(img_data)
        age_info = outs[0]
        gender_info = outs[1]
        return age_info, gender_info

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits))
        out = exp / np.sum(exp)
        return out

    def pred_img_with_boxes(self, img, boxes, landmarks):
        img = img_utils.decode_img(img)
        img_h, img_w = img.shape[:2]
        face_infos = []
        for box,landmark in zip(boxes, landmarks):
            e_box = box_utils.expand_box1(box[:4], img_w, img_h, expand_ratio=1.1)
            e_x1, e_y1, e_x3, e_y3 = e_box[:4]
            crop_img = img[e_y1:e_y3, e_x1:e_x3]
            aligned_img = self.align_img(crop_img, landmark["lmk_info"]["lmk_xy"])
            age_info, gender_info = self.pred_img(img=aligned_img)
            gender_prob = self.softmax(gender_info.squeeze())
            age_prob = self.softmax(age_info.squeeze())
            if self.rgb_input==False:
                rank = np.array([i for i in range(80)])
                predicted_age = np.sum(rank*age_prob)
            else:
                rank = np.array([i for i in range(6)]) # 年龄阶段
                predicted_age = np.sum(rank*age_prob)

            face_info = dict(
                box=box,
                age_info=predicted_age,
                gender_info=gender_prob
            )
            face_infos.append(face_info)
        return face_infos


if __name__ == '__main__':
    onnx_path = '/data/bff/sd_libs/conditions/detectors/pipfacelmk106_exp20_uncertainv1_epoch498_size112_sim.onnx'
    demo = FaceAgeEstimator(onnx_path=onnx_path)
    img_path = '/data/bff/data/face/face3.jpg'
    demo.pred_img(img_path)
