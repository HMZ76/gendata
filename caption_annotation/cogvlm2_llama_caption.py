import io
import os
import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

# need transformers 4.43.0

TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

class CaptionAnnotationVideo:
    def __init__(self, model_path="/home/nas01/xiongwenbo/cogvlm2-llama3-caption"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=TORCH_TYPE, device_map="auto", trust_remote_code=True)


    def load_video(self, video_data, strategy='chat'):
        bridge.set_bridge('torch')
        mp4_stream = video_data
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames,
                            int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break

        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data
    
    def predict(self, prompt, video_data, temperature):
        strategy = 'chat'

        video = self.load_video(video_data, strategy=strategy)

        history = []
        query = prompt
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    
    def run(self, video_path):
        prompt = "Please describe this video in detail."
        temperature = 0.1
        video_data = open(video_path, 'rb').read()
        response = self.predict(prompt, video_data, temperature)

        return response

if __name__ == '__main__':
    # test()
    runner = CaptionAnnotationVideo()

    image_folder = "image"
    
    ls = os.listdir(image_folder)
    ls = [i for i in ls if i.endswith('.mp4')]

    for image_path in ls:
       print(image_path)
       response = runner.run(os.path.join(image_folder,image_path))
       print(response)
