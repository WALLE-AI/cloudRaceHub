import loguru
from openai import OpenAI
import os
from dotenv import load_dotenv

from models.llm import LLMApi
load_dotenv()
def get_response():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="qwen2-7b-instruct",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你是谁？'}]
        )
    print(completion.model_dump_json())
    
    
def test_call_llm():
    query ="你是谁"
    prompt = LLMApi.build_text_prompt(query)
    llm_type = "tongyi"
    model_name = "qwen2-7b-instruct"
    response = LLMApi.call_llm(prompt=prompt,llm_type=llm_type,model_name=model_name)
    loguru.logger.info(f"response:{response}")

if __name__ == '__main__':
    test_call_llm()