from moviepy.editor import VideoFileClip

class QualityDetect:
    def __init__(self):
        pass

    def __call__(self, video_path):
        clip = VideoFileClip(video_path)
        width = clip.size[0]  # 宽
        height = clip.size[1]  # 高
        duration = clip.duration  # 视频时长（秒）
        framerate = clip.fps  # 视频帧率

        return {
            'width': width,
            'height': height,
            'duration': duration,
            'framerate': framerate,
        }


if __name__ == '__main__':
    q = QualityDetect()
    res = q("/home/nas01/wangqiteng/data/VidGen/Fudan-FUXI/video_clip_2/vidgen/VidGen_video_1481/klzZEwEFB5s-Scene-0314_1.mp4")
    print(res)
