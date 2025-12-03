import os
import time
import random
from moviepy.video.io.VideoFileClip import VideoFileClip
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def timecode_to_seconds(timecode):
    """
    将时间码转换为秒数。

    :param timecode: 时间码（格式为 HH:MM:SS.mmm）
    :return: 对应的秒数
    """
    dt = datetime.strptime(timecode, '%H:%M:%S.%f')
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


class VideoSceneDetect:
    def __init__(self, threshold=27, min_scene_len=15):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def split_video_segment(self, video_path, start_time, end_time, output_folder, base_name, idx, ext):
        output_name = f"{base_name}_{idx}{ext}"
        output_path = os.path.join(output_folder, output_name)
        clip = VideoFileClip(video_path).subclip(start_time, end_time)
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f'Scene {idx} saved as {output_path}')

        return output_path
    
    def split_video_by_timestamps(self, output, video_path, timestamps, duration_thres=8):
        base_name, ext = os.path.splitext(os.path.basename(video_path))
        splited_videos = []

        # Create a folder for the video splits
        if not os.path.exists(output):
            os.makedirs(output)

        with ThreadPoolExecutor() as executor:
            future_to_idx = {}
            for idx, (start_time, end_time) in enumerate(timestamps, 1):
                scene_duration = end_time - start_time
                if scene_duration < duration_thres:
                    continue

                future = executor.submit(
                    self.split_video_segment,
                    video_path,
                    start_time,
                    end_time,
                    output,
                    base_name,
                    idx,
                    ext
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                result = future.result()
                if result:
                    splited_videos.append(result)

        return splited_videos
    
    def detect_scenes(self, video_path, clip_duration):
        """
        使用PySceneDetect检测视频中的场景，返回场景时间戳列表。

        :param video_path: 视频文件路径
        :return: 场景时间戳列表
        """
        # 打开视频文件
        video = open_video(video_path)

        scene_manager = SceneManager()
        content_detector = ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
        scene_manager.add_detector(content_detector)

        # 对视频进行场景检测
        scene_manager.detect_scenes(video)

        # 获取检测到的场景列表
        scene_list = scene_manager.get_scene_list()

	
        # 构建时间戳列表
        timestamps = []
        if scene_list:
            for scene in scene_list:
                start_time = scene[0].get_timecode()
                end_time = scene[1].get_timecode()
                timestamps.append((timecode_to_seconds(start_time)+0.05, timecode_to_seconds(end_time)-0.05))
                # print(f'Detected scene from {start_time} to {end_time}')
        else:
            # 如果没有检测到场景切换，则将整个视频按照8-10秒随机间隔进行切分
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                start_time = 0
                while start_time < duration:
                    random_interval = random.uniform(clip_duration[0], clip_duration[1])
                    end_time = min(start_time + random_interval, duration)
                    timestamps.append((start_time, end_time))
                    start_time = end_time

        return timestamps
    
    def run(self, video_path, output, clip_duration, duration_thres):
        # 检测视频中的场景并获取时间戳
        timestamps = self.detect_scenes(video_path, clip_duration)

        # 根据时间戳切割视频
        splited_videos = self.split_video_by_timestamps(output, video_path, timestamps, duration_thres)
        
        return splited_videos
