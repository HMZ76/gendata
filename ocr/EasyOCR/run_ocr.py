import json
from easyocr import Reader


class OCRRunner():
    def __init__(self, detail):
        self.detail = detail
        self.reader = Reader(['ch_sim','en']) # 加载模型

    def convert_dtpye(self, ocr_result):
        for data in ocr_result:
            data['boxes'] = [[int(item) for item in sublist] for sublist in data['boxes']]

        return ocr_result

    def run(self, image):
        ocr_result = self.reader.readtext(image=image, detail=self.detail, output_format='dict')
        ocr_result = self.convert_dtpye(ocr_result)

        return ocr_result

if __name__ == '__main__':
    ocr_runner = OCRRunner(detail=True)
    ocr_result = ocr_runner.run(image="./examples/chinese.jpg")
    tmp = {
        'ocr': ocr_result
    }
    json.dumps(tmp, ensure_ascii=False, indent=4)
    print(ocr_result)