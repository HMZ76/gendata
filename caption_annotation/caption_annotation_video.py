from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForCausalLM 
from qwen_vl_utils import process_vision_info
import torch
import os, json, time, logging


class CaptionAnnotationVideo:
    def __init__(self, enable_en=True, enable_zh=False, model_path="Qwen2-VL-72B-Instruct", llm_model_path="Qwen2.5-7B-Instruct"):
        self.min_pixels = 256*28*28
        self.max_pixels = 1280*384*384
        self.enable_en = enable_en
        self.enable_zh = enable_zh
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        
        self.runner = None

    def caption_video(self,video_path, query):
        messages = [
            {"role": "user", 
             "content": [
                {"type": "text", "text": query},
                {"type": "video",
                 "video": video_path,
                 "fps": 1.0,
                 "max_pixels": 360*420,
                 }
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        
        res = ""
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = [r.strip() for r in response]

            for res_ in response:
                res += res_ + "\n"
        
        return res
    
    def caption_picture(self, image_path):
        query = "most briefly describe reasonable, slight and smooth movement in this image."
        messages = [
            {
                "role": "user",
                "content": [

                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        res = ""
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = [r.strip() for r in response]

            for res_ in response:
                res += res_

        return res


    def run_llm(self, caption, enable_en=True):
        prompt = f"""
        You are a helpful assistant.
        Please merge the following descriptions into a complete paragraph without adding any extra statements and keeping the content unchanged: {caption}
        """

        chinese_prompt = f"""
        你是一个助手。
        请将以下描述融合成一段完整的表述, 不要添加多余的语句, 内容上保持原状：{caption}
        """

        if enable_en:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "你是Qwen, 由阿里巴巴云提供支持。"},
                {"role": "user", "content": chinese_prompt}
            ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
             generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=512
               )
             generated_ids = [
               output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
             ]

             response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def run(self, video_path):
        
        query1 = "I am preparing to label video data for training a video generation model. Please describe this video from the perspective of a video creator in detail, including: \
        1. Subject: The subject is the main object in the video and an important embodiment of the theme. It can be a person, animal, plant, object, etc. \
        2. Subject description: The appearance details and posture of the subject, which can be described in multiple short sentences. It includes the movement performance, hairstyle and color, clothing,     facial features, body posture, etc. \
        3. Subject movement: The description of the subject's movement status, including stillness and motion. The movement status should not be too complex. \
        4. Scene: The scene is the environment in which the subject is located, including foreground and background. \
        5. Scene description: The detailed description of the environment in which the subject is located. It can be described in multiple short sentences, but should not be too excessive. It includes in    door scene, outdoor scene, natural scene, etc. \
        6. Camera language: Refers to the use of various applications of the lens, as well as the connection and switching between lenses to convey the story or information, and create a specific visual     effect and emotional atmosphere. For example, ultra-long shot, background blur, close-up, telephoto lens shooting, ground shooting, top shooting, aerial photography, depth of field, etc. \
        7. Light and shadow: Light and shadow are the key elements that give photography works a soul. The use of light and shadow can make the photo more profound and emotional, and create works with ri    ch sense of hierarchy and emotional expression. For example, ambient lighting, morning light, sunset, light and shadow, ding-dar effect, lighting, etc. \
        8. Atmosphere: The description of the expected atmosphere of the image, such as a lively scene, movie-level color grading, warm and beautiful, etc. \
Please describe the above 8 points as concisely as possible."
        
        #query2 = "most briefly describe reasonable, slight and smooth movement in this image."
        
        chinese_query1 = "我准备对视频数据进行标注，以用于训练视频生成模型。请站在视频创作者的角度详细描述这个视频。包括 \
        1.主体：主体是视频中的主要表现对象，是画面主题的重要体现者。如人、动物、植物，以及物体等；\
        2.主体描述：对主体外貌细节和肢体姿态等的描述，可通过多个短句进行列举。如运动表现、发型发色、服饰穿搭、五官形态、肢体姿态等；\
        3.主体运动：对主体运动状态的描述，包括静止和运动等，运动状态不宜过于复杂，符合5s视频内可以展现的画面即可；\
        4.场景：场景是主体所处的环境，包括前景、背景等；\
        5.场景描述：对主体所处环境的细节描述，可通过多个短句进行列举，但不宜过多，符合5s视频内可以展现的画面即可。如室内场景、室外场景、自然场景等。\
        6.镜头语言：是指通过镜头的各种应用以及镜头之间的衔接和切换来传达故事或信息，并创造出特定的视觉效果和情感氛围。如超大远景拍摄，背景虚化、特写、\
        长焦镜头拍摄、地面拍摄、顶部拍摄、航拍、景深等；（注意：这里与运镜控制作区分）\
        7.光影：光影是赋予摄影作品灵魂的关键元素，光影的运用可以使照片更具深度，更具情感，我们可以通过光影创造出富有层次感和情感表达力的作品。如氛围光\
        照、晨光、夕阳、光影、丁达尔效应、灯光等；\
        8.氛围：对预期视频画面的氛围描述。如热闹的场景、电影级调色、温馨美好等。\
        上述8条内容均须描述，尽可能精炼、完整回答。"
        
        #chinese_query2 = "我准备对视频数据进行标注，以用于训练视频生成模型。请站在视频创作者的角度详细描述这个视频。包括 \
        #1.运镜方式：运镜方式拍摄视频时候采用的镜头移动方式，如前推，后拉，平移，摇镜，旋转或没有明显的运镜手法等；\
        #2.镜头移动的方向：指的是拍摄时镜头移动的方向，如从下到上，从右到左，向前，向后，从上到下，从左到右，没有移动等；\
        #3.镜头移动的速度：指的是拍摄视频时镜头移动的速度，如快速，适中，缓慢，无移动，注意我们认为航拍的速度为快速；\
        #注意：航拍一般都是向前推进镜头，没有上下移动。\
       #上述3条内容均须描述，尽可能精炼。完整回答。"
        
        
        chinese_query2 = "请对视频拍摄镜头视角和运镜手法做出描述"
        #chinese_query2 = "请对视频拍摄时候的镜头移动方式进行描述"
        result = {}

        if self.enable_en:
            caption = self.caption_video(video_path, query1)
            res = self.run_llm(caption, True)
            result["en"] = res
        
        if self.enable_zh:
            caption1 = self.caption_video(video_path, chinese_query1)
            res1 = self.run_llm(caption1, False)
            caption2 = self.caption_video(video_path, chinese_query2)
            res2 = self.run_llm(caption2,False)
            result["zh"] = res1
        
        return result
    

if __name__ == "__main__":
   

    '''
    import time 
    runner = CaptionAnnotationVideo(model_path="/home/nas01/wangqiteng/model/qwen/Qwen2-VL-7B-Instruct", llm_model_path="/home/nas01/wangqiteng/model/qwen/Qwen2.5-3B-Instruct")
    
    image_folder = "image"
    ls = os.listdir(image_folder)
    ls = [i for i in ls if i.endswith('.mp4')]
    ls = sorted(ls,key=lambda x:int(x.split('.')[0]))
    start = time.time()
    for image_path in ls:
        print(image_path) 
        caption = runner.run(os.path.join(image_folder, image_path))
        print(caption) 

    end = time.time() 
    print(end-start)
    '''
    
    runner = CaptionAnnotationVideo(model_path="/home/nas01/wangqiteng/model/qwen/Qwen2-VL-7B-Instruct", llm_model_path="/home/nas01/wangqiteng/model/qwen/Qwen2.5-3B-Instruct")
    runner.run("/home/nas01/CVG_WH_VIDEO_DATASET/cartoon_1203/high_quality_video/10、纯白花嫁_1_part2.mp4")
