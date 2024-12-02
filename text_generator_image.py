from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os
from dotenv import load_dotenv
load_dotenv()

# prompt = "近景镜头，18岁的中国女孩，古代服饰，圆脸，正面看着镜头，民族优雅的服装，商业摄影，室外，电影级光照，半身特写，精致的淡妆，锐利的边缘。"

def execute_image_gen(prompt):
    rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"),
                          model=ImageSynthesis.Models.wanx_v1,
                          prompt=prompt,
                          n=1,
                          style='<watercolor>',
                          size='1024*1024')
    print('response: %s' % rsp)
    if rsp.status_code == HTTPStatus.OK:
        # 在当前目录下保存图片
        for result in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open('./%s' % file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))