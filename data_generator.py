import os
import hashlib
import json
import time
import logging
import shutil

from tqdm import tqdm
from multiprocessing import Pool
from utils.common import load_config, get_input_list, get_args
from concurrent import futures

from utils.common import get_keyframes


class DataGeneratior:
    def __init__(self, config):
        self.config = config
        self.log_dir = config["log_dir"]
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.all_logger = self.logger_config(os.path.join(config["log_dir"], "all_info.log"), "all_info")
        self.filter_logger = self.logger_config(os.path.join(config["log_dir"], "filter_info.log"), "filter_info")
        self.quality_detect_runner = None
        self.scene_detect_runner = None
        self.ocr_runner = None
        self.evaluate_runner = None
        self.optical_flow_runner = None
        self.classify_runner = None
        self.caption_runner = None
        self.load_models()

    def logger_config(self, log_dir, log_name):
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_dir, encoding='UTF-8')
        fh.setLevel(logging.INFO)
        ch = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(ch)
        logger.addHandler(fh)

        return logger

    def load_models(self):

        if self.config["quality_detect"]["implement"]:
            from quality_detect.quality_detecter import QualityDetect
            self.quality_detect_runner = QualityDetect()

        if self.config["video_scene_detect"]["implement"]:
            from video_scene_detect.split_video import VideoSceneDetect
            self.scene_detect_runner = VideoSceneDetect()

        if self.config["ocr"]["implement"]:
            from ocr.EasyOCR.run_ocr import OCRRunner
            self.ocr_runner = OCRRunner(detail=self.config["ocr"]["detail"])

        if self.config["video_evaluate"]["implement"]:
            from video_evaluate.DOVER.run_evaluate import EvaluateRunner
            self.evaluate_runner = EvaluateRunner(opt_path=self.config["video_evaluate"]["opt"])

        if self.config["optical_flow"]["implement"]:
            from optical_flow.run_optical_flow import OpticalFlowRunner
            self.optical_flow_runner = OpticalFlowRunner(threshold=self.config["optical_flow"]["threshold"])
        
        if self.config["classification"]["implement"]:
            from classify.classifier import Classifier
            self.classify_runner = Classifier(num_classes=self.config["classification"]["num_classes"], img_size=self.config["classification"]["img_size"])

        if self.config["caption_annotation"]["implement"]:
            if self.config["caption_annotation"]["caption_method"] == "cogvlm2":
                from caption_annotation.cogvlm2_llama_caption import CaptionAnnotationVideo
                self.caption_runner = CaptionAnnotationVideo(model_path=self.config["caption_annotation"]["model_path"])
            elif self.config["caption_annotation"]["caption_method"] == "qwen2":
                from caption_annotation.caption_annotation_video import CaptionAnnotationVideo
                self.caption_runner = CaptionAnnotationVideo(enable_en=self.config["caption_annotation"]["enable_en"], enable_zh=self.config["caption_annotation"]["enable_zh"],model_path=self.config["caption_annotation"]["model_path"], llm_model_path=self.config["caption_annotation"]["llm_model_path"])


    def create_annotation(self):
        annotation = {
            "video_name": None,
            "sha256": None,
            "width": None,
            "height": None,
            "fps": None,
            "duration": None,
            "optical_flow_score":None,
            "aesthetic_score":None,
            "caption_cogvlm2": None,
            "caption_qwen2": None,
            "caption_qwen2_zh": None,
            "category": None,
            "category_confidence_level":None,
            "caption":None
        }

        return annotation

    def get_sha256(self, clip_path):
        start_time = time.time()
        sha256_hash = hashlib.sha256()

        with open(clip_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        sha256_hash = sha256_hash.hexdigest()
        end_time = time.time()
        print(f"[INFO]: get_sha256 time used: {end_time - start_time}")

        return True

    def get_quality(self, clip_path):
        try:
            if self.quality_detect_runner is not None:

                start_time = time.time()
                
                video_properties = self.quality_detect_runner(clip_path)
                width = video_properties["width"]
                height = video_properties["height"]
                fps = video_properties["framerate"]
                duration = video_properties["duration"]
                end_time = time.time()
                print(f"[INFO]: get_quality time used: {end_time - start_time}")

                res = {}
                res["width"] = width
                res["height"] = height
                res["fps"] = fps
                res["duration"] = duration

                if width*height < self.config["quality_detect"]["height"]*self.config["quality_detect"]["width"] or \
                    fps < self.config["quality_detect"]["fps"] or duration < self.config["quality_detect"]["duration"] or duration>18:
                    res["low_quality"] = True
                else:
                    res["low_quality"] = False
        
                return res
            else:
                return None
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")

    def get_ocr(self, image):
        '''
        [
            {'boxes': [[224, 55], [461, 55], [461, 122], [224, 122]], 'text': '高臧左瞥站', 'confident': 0.016127214619305545}, 
            {'boxes': [[228, 107], [317, 107], [317, 147], [228, 147]], 'text': 'HSR', 'confident': 0.9962534611163719}, 
            {'boxes': [[330, 112], [465, 112], [465, 152], [330, 152]], 'text': 'Station', 'confident': 0.9995685954181702}, 
            {'boxes': [[154, 151], [471, 151], [471, 225], [154, 225]], 'text': '汽事酶停接途匾', 'confident': 0.009757362020189526}, 
            {'boxes': [[163, 209], [229, 209], [229, 245], [163, 245]], 'text': 'Car', 'confident': 0.9918220349097616}, 
            {'boxes': [[240, 210], [387, 210], [387, 249], [240, 249]], 'text': 'Kiss and', 'confident': 0.8391310095260222}, 
            {'boxes': [[397, 215], [473, 215], [473, 251], [397, 251]], 'text': 'Ride', 'confident': 0.9992946982383728}
        ]
        '''
        try:
            tag = False
            if self.ocr_runner is not None:
                tag = False
                time_start = time.time()
                ocr_result = self.ocr_runner.run(image)
                total_area = 0
                for res in ocr_result:
                    boxes = res["boxes"]
                    left_bottom, right_bottom, right_top, _ = boxes
                    width = right_bottom[0] - left_bottom[0]
                    height = right_top[1] - right_bottom[1]
                    area = width * height
                    total_area += area

                ratio = total_area / (self.annotation['width'] * self.annotation['height'])

                if ratio >= self.config['ocr']['threshold']:
                    tag = True

                end_time = time.time()
                print(f"[INFO]: get_ocr time used: {end_time - time_start}")

            return tag
            
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")

    def get_score(self, clip_path: str):
        try:
            if self.evaluate_runner is not None:
                start_time = time.time()
                fuse_results = self.evaluate_runner.run(clip_path=clip_path)
                end_time = time.time()
                print(f"[INFO]: get_score time used: {end_time - start_time}")

                return fuse_results
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")

    def get_optical_flow(self, clip_path: str):
        try:
            if self.optical_flow_runner is not None:
                start_time = time.time()
                tag, optical_flow_score = self.optical_flow_runner.run(clip_path=clip_path)
                end_time = time.time()
                print(f"[INFO]: get_optical_flow time used: {end_time - start_time}")

                return tag, optical_flow_score
            return False, 0
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")
            return False, 0

    def get_video_category(self, image):
        try:
            if self.classify_runner is not None:
                start_time = time.time()
                category = self.classify_runner(image)
                end_time = time.time()
                print(f"[INFO]: get_category time used: {end_time - start_time}")
                return category
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")

    def get_caption(self, video_path):
        try:
            if self.caption_runner is not None:
                start_time = time.time()
                caption = self.caption_runner.run(video_path)
                end_time = time.time()
                print(f"[INFO]: get_caption time used: {end_time - start_time}")
                return caption
        except Exception as e:
            self.all_logger.info(f"[ERROR]: {e}")

    def save_annotation(self, clip_path):
        annotation_path = clip_path.replace(".mp4", ".json")
        if not os.path.exists(annotation_path):
             with open(annotation_path, "w") as f:
                  f.write(json.dumps(self.annotation, indent=4, ensure_ascii=False))
        else:
             with open(annotation_path,'r') as f:
                  j = json.load(f)
                  j['caption_cogvlm2'] = self.annotation['caption_cogvlm2']
             with open(annotation_path,'w') as f:
                  f.write(json.dumps(j, indent=4, ensure_ascii=False))
                  

    def process(self, input_list):

        self.all_logger.info(f"start processing {len(input_list)} videos")
        length = len(input_list)
        startIndex = self.config['startIndex']

        output_dir = self.config['output']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        temp_dir = os.path.join(output_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        #with open("../final_anno1.json",'r') as f:
         #    l = json.load(f)[4700:]
        
        
        for index in range(startIndex, length):
            video_path = input_list[index]
            
            self.all_logger.info(f"processing the {index} video, name is {video_path}")

            # 1. filter low resolution video
            if self.quality_detect_runner is not None:
                res = self.get_quality(video_path)

                if res is not None and res["low_quality"] == True:
                    self.filter_logger.info(f"{video_path} is low resolution video. resulotion: {res['width']} x {res['height']}, fps: {res['fps']}, duration: {res['duration']}s")
                    continue
            
            # 2. video scene split
            if self.scene_detect_runner is not None:
                try:
                    splited_video = self.scene_detect_runner.run(video_path=video_path, output=temp_dir, clip_duration = [8, 10], duration_thres=4)
                except Exception as e:
                    self.all_logger.error(f"scene detect error: {e}")
                    continue
            else:
                if self.config["is_replace"]:
                    splited_video = [video_path]
                else:
                    splited_video = [os.path.join(temp_dir, os.path.basename(video_path))]
                    shutil.copyfile(video_path, os.path.join(temp_dir, os.path.basename(video_path)))

            dubious_dir = os.path.join(output_dir, "dubious_video")
            if not os.path.exists(dubious_dir):
                os.makedirs(dubious_dir)

            print(splited_video)
            for split_video in splited_video:
                # 2.5 create annotation
                self.annotation = self.create_annotation()

                # 3. get keyframes
                try:
                    keyframes, video_info = get_keyframes(clip_path=split_video)
                    self.annotation["width"] = video_info["width"]
                    self.annotation["height"] = video_info["height"]
                    self.annotation["fps"] = video_info["fps"]
                    self.annotation["duration"] = video_info["duration"]
                except Exception as e:
                    self.filter_logger.error(f"get keyframes error: {e}")
                    break
                

                # 4. get ocr result
                if self.ocr_runner is not None:
                    ocr_result = self.get_ocr(image=keyframes["first_frame"])
                    if ocr_result:
                      try:
                        shutil.move(split_video, dubious_dir)
                        self.filter_logger.info(f"{split_video} has too many ocr result")
                      except:
                        print()
                      continue

                # 5. get optical flow
                if self.optical_flow_runner is not None:
                    is_motion, optical_flow_score = self.get_optical_flow(split_video)
                    if is_motion == False:
                        try:
                           shutil.move(split_video, dubious_dir)
                           self.filter_logger.info(f"{split_video} is still video")
                        except:
                           print()
                        continue
                    self.annotation['optical_flow_score'] = optical_flow_score

                # 6. get video aesthetic evaluation
                if self.evaluate_runner is not None:
                    video_evaluation = self.get_score(clip_path=split_video)
                    if video_evaluation["overall"] < self.config["video_evaluate"]["threshold"]:
                        try:
                           shutil.move(split_video, dubious_dir)
                           self.filter_logger.info(f"{split_video} is low aesthetic quality")
                        except:
                           print()
                        continue
                    self.annotation['aesthetic_score'] = video_evaluation

                # 7. get video category
                if self.classify_runner is not None:
                    video_category, prob = self.get_video_category(keyframes["first_frame"])
                    save_dir = os.path.join(self.config["output"], video_category)
                    self.annotation['category'] = video_category
                    self.annotation["category_confidence_level"] = prob
                else:
                    save_dir = self.config["output"]

                # 8. get caption
                if self.caption_runner is not None:
                    caption = self.get_caption(split_video)
                else:
                    caption = {"en": "", "zh": ""}

                 # 9. save video
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # shutil.move(split_video, save_dir)
                
                #self.annotation['caption'] = l[index]['cap']
                # 10. save json
                if self.config["save_annotation"]:
                  try:
                    self.annotation["video_name"] = os.path.basename(split_video)
                    if self.config["caption_annotation"]["caption_method"] == "qwen2":
                        self.annotation["caption_qwen2"] = caption["en"]
                        #self.annotation["caption_qwen2_zh"] = caption["zh"]
                    elif self.config["caption_annotation"]["caption_method"] == "cogvlm2":
                        self.annotation["caption_cogvlm2"] = caption
                    if self.config["is_replace"]:
                        self.save_annotation(split_video)
                    else:
                        self.save_annotation(os.path.join(save_dir, self.annotation["video_name"]))
                  except:
                      continue
         
