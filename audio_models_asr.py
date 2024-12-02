import os
import re
import json
from urllib import request
from http import HTTPStatus

import dashscope

# 将your-dashscope-api-key替换成您自己的API-KEY
dashscope.api_key =os.environ['API_KEY']


def parse_sensevoice_result(data, keep_trans=True, keep_emotions=True, keep_events=True):
    '''
    本工具用于解析 sensevoice 识别结果
    keep_trans: 是否保留转写文本，默认为True
    keep_emotions: 是否保留情感标签，默认为True
    keep_events: 是否保留事件标签，默认为True
    '''
    # 定义要保留的标签
    emotion_list = ['NEUTRAL', 'HAPPY', 'ANGRY', 'SAD']
    event_list = ['Speech', 'Applause', 'BGM', 'Laughter']

    # 所有支持的标签
    all_tags = ['Speech', 'Applause', 'BGM', 'Laughter',
                'NEUTRAL', 'HAPPY', 'ANGRY', 'SAD', 'SPECIAL_TOKEN_1']
    tags_to_cleanup = []
    for tag in all_tags:
        tags_to_cleanup.append(f'<|{tag}|> ')
        tags_to_cleanup.append(f'<|/{tag}|>')
        tags_to_cleanup.append(f'<|{tag}|>')

    def get_clean_text(text: str):
        for tag in tags_to_cleanup:
            text = text.replace(tag, '')
        pattern = r"\s{2,}"
        text = re.sub(pattern, " ", text).strip()
        return text

    for item in data['transcripts']:
        for sentence in item['sentences']:
            if keep_emotions:
                # 提取 emotion
                emotions_pattern = r'<\|(' + '|'.join(emotion_list) + r')\|>'
                emotions = re.findall(emotions_pattern, sentence['text'])
                sentence['emotion'] = list(set(emotions))
                if not sentence['emotion']:
                    sentence.pop('emotion', None)

            if keep_events:
                # 提取 event
                events_pattern = r'<\|(' + '|'.join(event_list) + r')\|>'
                events = re.findall(events_pattern, sentence['text'])
                sentence['event'] = list(set(events))
                if not sentence['event']:
                    sentence.pop('event', None)

            if keep_trans:
                # 提取纯文本
                sentence['text'] = get_clean_text(sentence['text'])
            else:
                sentence.pop('text', None)

        if keep_trans:
            item['text'] = get_clean_text(item['text'])
        else:
            item.pop('text', None)
        item['sentences'] = list(filter(lambda x: 'text' in x or 'emotion' in x or 'event' in x, item['sentences']))
    return data

def execute_asr_models():
    task_response = dashscope.audio.asr.Transcription.async_call(
        model='sensevoice-v1',
        file_urls=[
            'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/sensevoice/rich_text_example_1.wav'],
        language_hints=['en'], )

    transcription_response = dashscope.audio.asr.Transcription.wait(
        task=task_response.output.task_id)

    if transcription_response.status_code == HTTPStatus.OK:
        for transcription in transcription_response.output['results']:
            url = transcription['transcription_url']
            result = json.loads(request.urlopen(url).read().decode('utf8'))
            print(json.dumps(parse_sensevoice_result(result, keep_trans=False, keep_emotions=False), indent=4,
                         ensure_ascii=False))
        print('transcription done!')
    else:
        print('Error: ', transcription_response.output.message)

if __name__ == "__main__":
    execute_asr_models()