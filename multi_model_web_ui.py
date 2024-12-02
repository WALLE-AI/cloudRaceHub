import os
import uuid
import numpy as np
from urllib3.exceptions import HTTPError
os.system('pip install dashscope  modelscope oss2 -U')

from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import oss2
import os
import re
import cv2
import os
import secrets
import tempfile
import requests
from http import HTTPStatus
from dashscope import MultiModalConversation
import dashscope
API_KEY = os.environ['API_KEY']
dashscope.api_key = API_KEY

REVISION = 'v1.0.4'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--revision", type=str, default=REVISION)
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=7860,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def extract_frames(video_path, output_folder,images_list,frame_interval=1, start_frame=0, end_frame=None,max_frames=None):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_id = 0
    extracted_frames = 0
    while True:
        # 读取视频的下一帧
        ret, frame = cap.read()
        
        # 如果正确读取帧，ret为True
        if not ret:
            break
        # 检查是否达到最大帧数限制
        if max_frames is not None and extracted_frames >= max_frames:
            break
        
        # 检查帧ID是否在指定的开始和结束帧之间，并且是否是指定的帧间隔的倍数
        if frame_id >= start_frame and (end_frame is None or frame_id <= end_frame) and (frame_id - start_frame) % frame_interval == 0:
        # 构建输出图片的文件名
            filename = os.path.join(output_folder, f"frame_{frame_id:04d}.png")
            images_list.append(filename)
        
            # 保存帧为图片
            cv2.imwrite(filename, frame)
            # 帧计数增加
            extracted_frames += 1
        
            # 帧ID递增
        frame_id += 1
    
    # 释放视频捕获对象
    cap.release()
    print(f"Extracted {frame_id} frames to {output_folder}")




def _launch_demo(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        if len(chat_query) == 0:
            _chatbot.pop()
            task_history.pop()
            return _chatbot
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""
        messages = []
        content = []
        for q, a in history_cp:
            if isinstance(q, (tuple, list)):
                if is_video_file(q[0]):
                    ##对视频进行逐帧解析
                    image_list = []
                    images_save = "data/images/"+str(uuid.uuid4())
                    extract_frames(f'file://{q[0]}',images_save,image_list,max_frames=5)
                    content.append({'video': image_list})
                else:
                    content.append({'image': f'file://{q[0]}'})
            else:
                content.append({'text': q})
                messages.append({'role': 'user', 'content': content})
                messages.append({'role': 'assistant', 'content': [{'text': a}]})
                content = []
        messages.pop()
        responses = MultiModalConversation.call(
            model='qwen-vl-max-latest', messages=messages, stream=True,
        )
        for response in responses:
            if not response.status_code == HTTPStatus.OK:
                raise HTTPError(f'response.code: {response.code}\nresponse.message: {response.message}')
            response = response.output.choices[0].message.content
            response_text = []
            for ele in response:
                if 'text' in ele:
                    response_text.append(ele['text'])
                elif 'box' in ele:
                    response_text.append(ele['box'])
            response_text = ''.join(response_text)
            _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(response_text))
            yield _chatbot

        if len(response) > 1:
            result_image = response[-1]['result_image']
            resp = requests.get(result_image)
            os.makedirs(uploaded_file_dir, exist_ok=True)
            name = f"tmp{secrets.token_hex(20)}.jpg"
            filename = os.path.join(uploaded_file_dir, name)
            with open(filename, 'wb') as f:
                f.write(resp.content)
            response = ''.join(r['box'] if 'box' in r else r['text'] for r in response[:-1])
            _chatbot.append((None, (filename,)))
        else:
            response = response[0]['text']
            _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        print("Qwen2-VL-Chat: " + _parse_text(full_response))
        yield _chatbot


    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        _chatbot_gen = predict(_chatbot, task_history)
        for _chatbot in _chatbot_gen:
            yield _chatbot

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=3>Cloud Race Multi Models。</center>""")

        chatbot = gr.Chatbot(label='Multi Models', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

    demo.queue().launch(
        share=args.share,
        # inbrowser=args.inbrowser,
        # server_port=args.server_port,
        # server_name=args.server_name,
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == '__main__':
    main()
