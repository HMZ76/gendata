import sys
import os
sys.path.append("..")
sys.path.append(os.getcwd())

import torch
from torchvision import transforms
from mobilenet_v3 import mobilenet_v3_large
from ultralytics.models import YOLO
from source.miface_api import MiFaceAPI
from PIL import Image
import cv2

MODEL_CKPT_PATH = {
    "detect": "classify/pretrained_models/yolov8n.pt",
    "classify": "classify/pretrained_models/mobilenetv3_classes_full_480.pth"
}

category_map = {
    0: "aurora",
    1: "building",
    2: "car",
    3: "flower",
    4: "indoor",
    5: "lightning",
    6: "nightscene",
    7: "rain_snow",
    8: "scenery",
    9: "starrysky",
    10: "cartoon",
    11: "food",
    12: "other",
    69: "person",
    70: "animal",
}

class Classifier(object):

    def __init__(self, img_size=480, num_classes=12, device=torch.device('cuda')):
        self.data_transform = transforms.Compose(
                        [transforms.Resize(img_size + 32),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.config = {
            'face_detector': True,
            'face_head_detector': False,
            'face_validator': False,
            'face_landmarker': False,
            'age_gender_identifier': False,
        }
        self.device = device
        
        self.face_detect = MiFaceAPI(onnx_device="gpu", config=self.config)
        self.detecter = YOLO(MODEL_CKPT_PATH["detect"])
        self.classify = mobilenet_v3_large(num_classes=num_classes)
        self.classify.load_state_dict(torch.load(MODEL_CKPT_PATH['classify'], map_location=device))
        self.classify.to(self.device).eval()

    
    def __call__(self, image):
        category = 0
        image = cv2.resize(image, (640, 640))
        image = Image.fromarray(image)

        results = self.face_detect(image)
        if len(results["face_boxes"]) > 0:
            boxes = results["face_boxes"]
            max_value = 0
            for box in boxes:
                if box[4] > max_value:
                    max_value = box[4]

            if max_value > 0.5:
                category = 69
                return category_map[category],max_value
        
        # run detect model
        class_ids = list(range(15, 24)) + [0]
        result = self.detecter.predict(source=image, imgsz=640, device="cuda", classes=class_ids, conf=0.5)
        cls_ids = result[0].boxes.cls.int().tolist()
        cls_ids = list(set(cls_ids))
        if len(cls_ids) > 0:
            if 0 in cls_ids:
                category = 69
            else:
                category = 70
            prob = float(torch.max(result[0].boxes.conf))
        else:
            
            inimg = self.data_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classify(inimg)
                output = output.squeeze(0).cpu()
                predict = torch.softmax(output, dim=0)
                topk = torch.topk(predict, k=1)
                category = topk.indices[0].item()
                prob = float(torch.max(predict)/torch.sum(predict))
        return category_map[category], prob
