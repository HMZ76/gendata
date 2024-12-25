import os, json
from utils.common import get_input_list, get_args, load_config
import shutil


def is_high_quality(j,config):
    width_thr = config['quality_detect']['width']
    height_thr = config['quality_detect']['height']
    fps_thr = config['quality_detect']['fps']
    duration_thr = config['quality_detect']['duration']
    aesthetic_score_thr = config['video_evaluate']['threshold']
    width = j['width']
    height = j['height']
    fps = j['fps']
    duration = j['duration']
    #optical_flow_score = j['optical_flow_score']
    #aesthetic_score = j['aesthetic_score']['aesthetic']
    low_optical_score = config['optical_flow']['low_bound']
    high_optical_score = config['optical_flow']['up_bound']
    #if(width*height<width_thr*height_thr or fps < fps_thr or duration < duration_thr or aesthetic_score < aesthetic_score_thr or optical_flow_score < low_optical_score or optical_flow_score > high_optical_score or duration > 12):
    if duration < duration_thr:
        return False
    else:
        return True

def filter_again(config):
    output_path = config['output']
    json_list = os.listdir(output_path)
    json_list = [i for i in json_list if i.endswith('.json')]
    if not os.path.exists(output_path+'/high_quality_json'):               
    	os.mkdir(output_path+'/high_quality_json')
    if not os.path.exists(output_path+'/high_quality_video'):               
        os.mkdir(output_path+'/high_quality_video')
    cnt = 0
    for j_path in json_list:
        with open(output_path+'/'+j_path,'r') as f:
            j = json.load(f)
            if is_high_quality(j,config):
                v_name = j['video_name']
                if os.path.exists(output_path+'/temp/'+v_name) and os.path.exists(output_path+'/'+j_path):
                  shutil.move(os.path.join(output_path,j_path), output_path+'/high_quality_json/'+j_path)
                  shutil.move(output_path+'/temp/'+v_name, output_path+'/high_quality_video/'+v_name)
            else:
                cnt += 1
    print(cnt)
if __name__ == '__main__':
   
   args = get_args()
   config = load_config(args.config)
   filter_again(config)
