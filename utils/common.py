import os
import cv2
import yaml
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    return args


def get_input_list(path):
    if path.endswith(".mp4"):
        return [path]
    elif os.path.isdir(path):
        return [os.path.join(path, i) for i in os.listdir(path)]
    elif path.endswith('.txt'):
        with open(path, 'r') as fr:
            return [i.strip() for i in fr.readlines()]
    else:
        print(f"file path is not supported! {path}")

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def get_keyframes(clip_path):
    # 打开视频文件
    cap = cv2.VideoCapture(clip_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None, None

    # 获取视频的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    # 获取首帧
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return None, None, None

    # 获取中间帧
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, middle_frame = cap.read()
    if not ret:
        print("Error: Could not read middle frame.")
        return first_frame, None, None

    # 获取最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if not ret:
        print("Error: Could not read last frame.")
        return first_frame, middle_frame, None

    # 释放视频文件
    cap.release()

    keyframes = {
        'first_frame': first_frame,
        'middle_frame': middle_frame,
        'last_frame': last_frame
    }

    video_info = {
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }

    return keyframes, video_info
