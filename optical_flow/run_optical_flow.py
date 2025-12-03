import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor


class OpticalFlowRunner():
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def process_frame(self, prev_gray, gray):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return np.mean(mag)

    def run(self, clip_path):
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # 每隔多少帧读取一次，达到1 fps的效果

        ret, pre_frame = cap.read()
        prev_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0
        results = []
        with ThreadPoolExecutor() as executor:
            while cap.isOpened():
                ret, cur_frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
                results.append(executor.submit(self.process_frame, prev_gray, gray))
                prev_gray = gray

        cap.release()
        flag = False
        optical_flow_score = 0
        for result in results:
            if result.result() > self.threshold:
                flag = True
                optical_flow_score += result.result()

        return flag, optical_flow_score/(len(results)+1)

if __name__ == '__main__':
    start_time = time.time()
    optical_runner = OpticalFlowRunner(threshold=1.0)
    rst = optical_runner.run(clip_path="/home/higher/hdisk/wangzepeng5/lm/GenData/video_scene_detect_project/output/v_ZZ71FIfxX-c_1/v_ZZ71FIfxX-c_1.mp4")
    end_time = time.time()
    print(f"[INFO]: 耗时 {end_time - start_time}s")
    print(rst)

