import os
import os.path as osp
import onnxruntime
import cv2
from PIL import Image
import numpy as np
from source.utils import utils, box_utils, img_utils

class FaceLandmarker(object):
    def __init__(self, onnx_path, device='cpu') -> None:
        self.onnx_path = onnx_path
        self.device = device

        self.build_model()
        self.input_nodes = self.session.get_inputs()
        self.output_nodes = self.session.get_outputs()

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
    
    def forward(self, batch_img):
        input_feed = {self.input_nodes[0].name : batch_img}
        output_names = [node.name for node in self.output_nodes]
        outputs = self.session.run(output_names, input_feed)
        return outputs

    def decode_batch(self, hm_pred:np.ndarray, offset_x_pred:np.ndarray, offset_y_pred:np.ndarray, vis_pred:np.ndarray, pose_pred:np.ndarray, uncertain_pred:np.ndarray):
        '''
        hm_pred                 s=(b,106,7,7)
        offset_x_pred           s=(b,106,7,7)
        offset_y_pred           s=(b,106,7,7)
        vis_pred                s=(b,106,7,7)
        pose_pred               s=(b,3,1,1)
        uncertain_pred          s=(b,1,1,1)
        '''
        batch_size = hm_pred.shape[0]

        assert hm_pred.ndim == 4 and hm_pred.shape[-1] == hm_pred.shape[-2]
        batch_size, num_lmks, feat_size, _ = hm_pred.shape
        #lmk_score
        hm_pred = hm_pred.reshape(batch_size, num_lmks, -1)#s=(b,106,7*7)
        hm_max_score_idx = np.argmax(hm_pred, axis=2)#s=(b,106) 7*7中最大值的索引
        lmk_score = hm_pred[np.arange(0, batch_size).reshape(batch_size, 1),np.arange(0, num_lmks).reshape(1, num_lmks), hm_max_score_idx]#s=(b,106)

        #lmk_vis_score
        vis_pred = vis_pred.reshape(batch_size, num_lmks, -1)#s=(b,106,7*7)
        lmk_vis_score = vis_pred[np.arange(0, batch_size).reshape(batch_size, 1), np.arange(0, num_lmks).reshape(1, num_lmks), hm_max_score_idx] * lmk_score#s=(b,106) 

        #lmk_xy
        hm_pred_sum = hm_pred.sum(-1)[..., np.newaxis]#s=(b,106,7*7)->(b,106)->(b,106,1)
        hm_pred = hm_pred / hm_pred_sum#s=(b,106,7*7)
        offset_x_pred = (offset_x_pred - 0.5) * 5
        offset_y_pred = (offset_y_pred - 0.5) * 5
        offset_x_pred = offset_x_pred.reshape(batch_size, num_lmks, -1)#s=(b,106,7*7)
        offset_y_pred = offset_y_pred.reshape(batch_size, num_lmks, -1)#s=(b,106,7*7)
        grid_x = np.arange(0, feat_size, 1, dtype=np.float32)
        grid_y = np.arange(0, feat_size, 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)#s=(7,7)  s=(7,7)

        offset_x_pred_rel_feat = offset_x_pred + grid_x.reshape(-1)#=s(b,106,7*7)
        offset_y_pred_rel_feat = offset_y_pred + grid_y.reshape(-1)#=s(b,106,7*7)
        lmk_x = np.sum(offset_x_pred_rel_feat * hm_pred, axis=-1)[..., np.newaxis] * self.stride
        lmk_y = np.sum(offset_y_pred_rel_feat * hm_pred, axis=-1)[..., np.newaxis] * self.stride
        lmk_xy = np.concatenate([lmk_x, lmk_y], axis=-1)#s=(b,106,2)
        
        #pose
        pose = pose_pred.squeeze(-1).squeeze(-1) * 180.0

        #uncertain
        uncertain = uncertain_pred.squeeze(-1).squeeze(-1).squeeze(-1)

        res = dict(
            batch_lmk_xy=lmk_xy,
            batch_lmk_score=lmk_score,
            batch_lmk_vis_score=lmk_vis_score,
            batch_pose=pose,
            batch_uncertain=uncertain,
        )
        return res
    
    def pred_img(self, img):
        """
        单张照片推理。由于需要和训练用图保持分布一致，不能直接传图使用。请使用with box的函数。
        """
        img_data, begin_x, begin_y, resize_w, resize_h, scale_w, scale_h = self.precess_img(img, size=112, keep_ratio=True, padding_val=0)
        outs = self.forward(img_data)
        res = self.decode_batch(*outs)
        res['batch_lmk_xy'] = (res['batch_lmk_xy'] - np.array([begin_x, begin_y], dtype=np.float32)) / np.array([scale_w, scale_h], dtype=np.float32) 
        lmk_xy = res['batch_lmk_xy'][0]
        lmk_score = res['batch_lmk_score'][0]
        lmk_vis_score = res['batch_lmk_vis_score'][0]
        pose = res['batch_pose'][0]
        uncertain = res['batch_uncertain'][0]

        lmk_info = dict(
            lmk_xy=lmk_xy,
            lmk_score=lmk_score,
            lmk_vis_score=lmk_vis_score,
            pose=pose,
            uncertain=uncertain
        )
        return lmk_info
    
    def pred_img_with_boxes(self, img, boxes):
        img = img_utils.decode_img(img)
        img_h, img_w = img.shape[:2]
        face_infos = []
        for box in boxes:
            e_box = box_utils.expand_box1(box[:4], img_w, img_h, expand_ratio=1.3)
            e_x1, e_y1, e_x3, e_y3 = e_box[:4]
            crop_img = img[e_y1:e_y3, e_x1:e_x3]
            lmk_info = self.pred_img(img=crop_img)
            lmk_info["lmk_xy"] += np.array([e_x1, e_y1], dtype=np.float32)#s=(106,2)
            face_info = dict(
                box=box,
                lmk_info=lmk_info
            )
            face_infos.append(face_info)
        return face_infos


if __name__ == '__main__':
    onnx_path = '/data/bff/sd_libs/conditions/detectors/pipfacelmk106_exp20_uncertainv1_epoch498_size112_sim.onnx'
    demo = FaceLandmarker(onnx_path=onnx_path)
    img_path = '/data/bff/data/face/face3.jpg'
    demo.pred_img(img_path)
