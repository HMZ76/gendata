import os
import sys
import json

sys.path.append("video_evaluate/DOVER")
sys.path.append("classify")
sys.path.append("optical_flow")
sys.path.append("caption_annotation")
sys.path.append("quality_detect")
sys.path.append("video_scene_detect")

import logging
import numpy as np
from utils.common import get_input_list, get_args, load_config
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)


def run(config, input_list):
    from data_generator import DataGeneratior
    generator = DataGeneratior(config)
    generator.process(input_list)


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config) # 加载配置文件
    input_list = sorted(get_input_list(config['input']))[900:]
    #ls = os.listdir(config['input'])
    #ls = sorted(ls, key=lambda x: int(x.split('_')[-1]))
    #input_list = []
    #for i in ls[1800:]:
    #    input_list.extend(get_input_list(os.path.join(config['input'], i)))


    #with open('../final_anno.json','r') as f:
     #    l = json.load(f)[4700:]
    #input_list = [j['path'] for j in l]
    #ls1 = os.listdir('/home/nas01/wangqiteng/data/pipeline_pet_video/dubious_video')
    #ls2 = os.listdir('/home/nas01/wangqiteng/data/pipeline_pet_video/temp') 
    #ls1 = list(set(['/home/nas01/xiongwenbo/pixabay_pet_video/'+i[:-6]+'.mp4' for i in ls1]))
    #ls2 = list(set(['/home/nas01/xiongwenbo/pixabay_pet_video/'+i[:-6]+'.mp4' for i in ls2]))
    #ls = list(set(ls1+ls2))
    #input_list = [i for i in input_list if i not in ls]
    run(config, input_list)
