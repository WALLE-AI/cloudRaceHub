import base64
from io import BytesIO
import os
from PIL import Image
import loguru
import requests
from streamlit import json
def pdf_file_image(pdf_file,zoomin=2):
    '''
    将pdf全部转成image
    '''
    import pdfplumber
    pdf = pdfplumber.open(pdf_file)
    images = [(i,p.to_image(resolution=72 * zoomin).annotated) for i, p in
                            enumerate(pdf.pages)]
    return images

def load_images_from_folder(folder_path):
    images_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images_list.append(filename)
    return images_list


def image_to_base64(image_path,root_path):
    root_path =root_path
    images_path_new = root_path + image_path
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        mime_type = image_path.split(".")[-1]
        with Image.open(images_path_new) as img:
            # 定义新的尺寸，例如缩小到原来的一半
            new_width = img.width // 2
            new_height = img.height // 2
            # 调整图片大小
            img_resized = img.resize((new_width, new_height))
            # 将图片转换为字节流
            buffered = BytesIO()
            img_resized.save(buffered, format=img.format)
            # 将字节流转换为Base64编码
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'data:image/{mime_type};base64,{img_base64}'

def pdf_image_to_base64(img):
    buffered = BytesIO()
    mime_type = "png"
    img.save(buffered, format=mime_type)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{mime_type};base64,{img_base64}'
    
    
def encode_image_base64_from_url(image_id, image_url):
    mime_type = image_id.split(".")[-1]
    try:
        # 发送GET请求获取图片内容
        response = requests.get(image_url)
        response.raise_for_status()  # 如果请求失败，这会抛出异常
        # 获取图片内容
        image_content = response.content
        # 将图片内容转换为base64编码
        base64_encoded = base64.b64encode(image_content).decode('utf-8')
        base64_encoded = f'data:image/{mime_type};base64,{base64_encoded}'
        return base64_encoded
    except requests.RequestException as e:
        print(f"download image error: {e}")
        return None
    except Exception as e:
        print(f"transformer process error: {e}")
        return None
    
def write_json_file_line(data_dict, save_file_name):
    with open(save_file_name, "w", encoding="utf-8") as file:
        for line in data_dict:
            file.write(json.dumps(line, ensure_ascii=False)+"\n")
         
def pdf_image_to_base64(img):
    buffered = BytesIO()
    mime_type = "png"
    img.save(buffered, format=mime_type)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{mime_type};base64,{img_base64}'


def download_image(url, filename):
    root_image = "data/sample10000_image/"
    images_dir_path = root_image + filename
    # loguru.logger.info(f"images dir path {images_dir_path}")
    if filename in load_images_from_folder(root_image):
        loguru.logger.info(f"image is exist,no donwload")
        return False
    else:
        try:
            response = requests.get(url)
            response.raise_for_status()  # 检查请求是否成功
            with open(images_dir_path, 'wb') as f:
                f.write(response.content)
                loguru.logger.info(f"image save：{filename}")
                return True
        except requests.RequestException as e:
            loguru.logger.info(f"request error：{e}")
            return False
        except IOError as e:
            loguru.logger.info(f"request io error：{e}")