import os
import os.path as osp
from typing import Any
import cv2
import numpy as np
import sys
sys.path.append(os.getcwd())
sys.path.append("../")
from source.face_detector import FaceDetector
from source.facehead_detector import FaceHeadDetector
from source.face_validator import FaceValidator
from source.face_landmarker import FaceLandmarker
from source.face_age_estimator import FaceAgeEstimator
from source.utils import utils, box_utils, img_utils

# config example. True means use default onnx file
config = {
    'face_detector': True,
    'face_head_detector': False,
    'face_validator': True,
    'face_landmarker': True,
    'age_gender_identifier': True,
}

class MiFaceAPI(object):
    def __init__(
            self,
            onnx_device:str="cpu",
            config:dict=None
    ):
        self._onnx_device = onnx_device
        self.set_model_path(config)
        self.build_modules()
    
    def set_model_path(self, config):
        # default path
        self.face_detector_path = "classify/onnx_models/facedetunion_exp15_epoch129_size480_sim.onnx"
        self.face_head_detector_path = "classify/onnx_models/facehead_det_exp13_epoch_merge_size480_sim.onnx"
        self.face_validator_path = "classify/onnx_models/facevalid_exp12_epoch462_size112_sim.onnx"
        self.face_landmarker_path = "classify/onnx_models/pipfacelmk106_exp20_uncertainv1_epoch498_size112_sim.onnx"
        self.age_gender_estimator_path = "classify/onnx_models/tinyage_gray_112_age_gender_sim.onnx"

        # config path
        if isinstance(config["face_detector"], str): self.face_detector_path = config["face_detector"]
        if isinstance(config["face_head_detector"], str): self.face_head_detector_path = config["face_head_detector"]
        if isinstance(config["face_validator"], str): self.face_validator_path = config["face_validator"]
        if isinstance(config["face_landmarker"], str): self.face_landmarker_path = config["face_landmarker"]
        if isinstance(config["age_gender_identifier"], str): self.age_gender_estimator_path = config["age_gender_identifier"]

        # if none
        if config["face_detector"]==False: self.face_detector_path = None
        if config["face_head_detector"]==False: self.face_head_detector_path = None
        if config["face_validator"]==False: self.face_validator_path = None
        if config["face_landmarker"]==False: self.face_landmarker_path = None
        if config["age_gender_identifier"]==False: self.age_gender_estimator_path = None

    def build_modules(self):
        assert self.face_detector_path is not None or self.face_head_detector_path is not None
        if self.face_detector_path:
            self.detector = FaceDetector(onnx_path=self.face_detector_path, device=self._onnx_device)
        if self.face_head_detector_path:
            self.detector = FaceHeadDetector(onnx_path=self.face_head_detector_path, device=self._onnx_device)
        if self.face_validator_path:
            self.validator = FaceValidator(onnx_path=self.face_validator_path, device=self._onnx_device)
        if self.face_landmarker_path:
            self.landmarker = FaceLandmarker(onnx_path=self.face_landmarker_path, device=self._onnx_device)
        if self.age_gender_estimator_path:
            assert self.face_landmarker_path is not None
            self.estimator = FaceAgeEstimator(onnx_path=self.age_gender_estimator_path, device=self._onnx_device)

    def __call__(self, img_path, vis=True, *args: Any, **kwds: Any) -> Any:

        results = dict(img_path=None,
                       face_boxes=None,
                       head_boxes=None,
                       face_validation_info=None,
                       face_landmark_info=None,
                       face_age_gender_info=None)
        results["img_path"] = img_path

        # step 1. detection
        img = img_utils.decode_img(img_path)
        if self.face_detector_path:
            face_boxes = self.detector.pred_img(img=img)
            results["face_boxes"] = face_boxes
        elif self.face_head_detector_path:
            face_boxes, head_boxes = self.detector.pred_img(img=img_path)
            results["face_boxes"] = face_boxes; results["head_boxes"] = head_boxes
        else:
            raise KeyError("No detector in config")
        
        # step 2. validation
        if self.face_validator_path:
            face_validation_info = self.validator.pred_img_with_boxes(img=img, boxes=results["face_boxes"])
            results["face_validation_info"] = face_validation_info
        
        # step 3. landmark
        if self.face_landmarker_path:
            face_landmark_info = self.landmarker.pred_img_with_boxes(img=img, boxes=results["face_boxes"])
            results["face_landmark_info"] = face_landmark_info
        
        # step 4. age and gender (need landmarks to align face images)
        if self.age_gender_estimator_path:
            face_age_gender_info = self.estimator.pred_img_with_boxes(img=img, 
                                                                      boxes=results["face_boxes"], 
                                                                      landmarks=results["face_landmark_info"])
            results["face_age_gender_info"] = face_age_gender_info

        return results
    
        