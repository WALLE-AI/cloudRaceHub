


import json

import loguru


data_example = {
}

def data_example_process():
    '''
    ##https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health 文本匹配的数据
      ## https://github.com/jeffheaton/app_generative_ai/tree/main
        training_data = [{"instruction": "判断以下医疗文本的科室分类","input": "患者出现头痛、恶心、视物模糊等症状","output": "神经内科"},# 更多训练样本...
    "conversations":[{"from":"human","value":"问：左踝关节扭伤5年未痊愈。2009年左踝扭伤至今已5年多，走路没多久就会酸疼，肿胀，如果走的时间稍>微长点，接下来的几天脚会很不舒服，酸疼，有些肿胀，早上起来肿胀会消退些。不走路的时候踝关节里也会有不舒服的感觉。2009年1月因在不平坦的路上走路崴了脚，当时没有疼痛，于是没有进行治疗，不久后
        跑步就出现左踝关节肿胀疼痛，敷了几剂药和泡了点中药后消了一点肿，疼痛也不明显，以为没有大碍，因为当时还在上学就没有再做进一步治疗。但是后来发现走路之后会酸痛肿胀，晚上会比较明显，早上起床>就会消点肿，休息过后肿痛会好转，于是又继续敷药，但也没什么起色。2009年9月，大学军训了一个星期左右，前几天要长期站着，后来因为脚疼也申请了休息。大学期间也免不了要做些运动，但还是尽量避免，
      发觉踝关节很容易就累，就酸痛，不能走很长时间。后来也做过理疗，针灸，但是效果不明显，只是感觉舒服一些。后来因为要出国读书就停止了治疗，平时也不做运动，尽量避免长时间走路。2012年7月做了核磁
      共振，到医院做了理疗，敷了药，"},{"from":"gpt","value":"答：作为医生，我建议您尽快前往医院进行进一步的检查和治疗。根据描述，您的左踝关节扭伤已经持续了5年，而且症状似乎没有得到缓解。这可能
      是由于初始扭伤时没有及时进行治疗，导致伤程加长并引起了慢性炎症。\n\n您之前尝试过一些治疗方法，如敷药、泡中药、理疗和针灸，但效果不明显。这可能是因为病因尚未明确，治疗方法不够全面和专业。>因此，我建议您前往医院接受全面的检查，可能包括X光、核磁共振或其他相关检查，以了解关节结构是否受损。\n\n一旦明确了病因，您可以与医生一起制定治疗计划。治疗可能包括物理治疗、康复训练、药物治
      疗或手术等，具体根据您的病情来定。同时，您也需要按医生建议进行生活方式调整，如适当休息、避免过度活动或长时间站立、做适当的运动来增强踝关节的稳定性。\n\n最重要的是，不要忽视疼痛和不适感，>及时就医是为了防止病情进一步恶化并提高痊愈的机会。祝您早日康复！"}]
    '''
    ##微调与不微调两种作为 
    # 1、一种使用通过API+prompt的方式完成对应NLP任务（NER 文本匹配文本 summary问题等等） 
    # 2、通过微调+prompt+rag（搜索api）的方式解决精度问题
    test_system_prompt = '''
    你是一个智能助手，你能够根据用户输入判断出文本的意图
    用户输入：{text}
    按如下json格式输出
    {
      "intent":""
    }
    '''
    system_prompt = "你是一个智能助手，能够高质量判断出文本的对话意图"
    data_dict_list = []
    for data  in data_example["data"]:
        data_dict = {}
        data_dict["callid"] = data['callid']
        data_dict["callid_no"] = data['callid_no']
        data_dict["label"] = data['label']
        data_dict["record_dt_info"] = data['record_dt_info']
        data_dict['对话角色']=data['对话角色']
        if data["对话角色"] == "CLIENT":
          ##这个任务明显就是多轮对话来判断对应意图表达
          _data_dict = {
            "instruction": system_prompt,
            "input":data["record_dt_info"],
            "output":data["label"]
          }
          data_dict['instruction_data'] =_data_dict
        else:
          data_dict["instruction_data"] = ""
        data_dict_list.append(data_dict)
    save_data = "data/example_test/test_intent.jsonl"
    with open(save_data,"w",encoding="utf-8") as file:
      file.write(json.dumps(data_dict_list,ensure_ascii=False,indent=4))
      

def dataset_text_ner_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        entity_names_list = set()
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            # match_names = ["地点", "人名", "地理实体", "组织"]
            match_names = ['银行名称', '银行', '形容词', '产品名称', '产品', '金融产品', '金融名词', '银行产品']
            
            entity_sentence = ""
            
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    entity_names_list.add(name)
                    if name in match_names:
                        entity_label = name
                        break
                
                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""
            
            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"
            
            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }
            
            messages.append(message)
    loguru.logger.info(f"name list {entity_names_list}")
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
            
            
def dataset_text_classfication_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    {
      "news id":"ID"
      "code":"CATEGORY CODE"
      "name":"CATEGORY NAME"
      "title":"TEXT"
      "keywords": "TEXT KEYWORDS"
      
    }
    news id:文本样本的唯一标识符code:文本所属的类别编码
    name:类别名称
    title:文本的标题
    keywords:文本的关键词
    """
    messages = []
    # 读取旧的JSONL文件
    with open(origin_path, "r",encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["keywords"]
            title = data["title"]
            label = data["name"]
            message = {
                "instruction": "你是一个文本分类领域的专家，你能够高质量判别出文本的类别，你会接收到一段文本和文本的主题，请输出文本内容的正确类别",
                "input": f"文本:{context},类型选型:{title}",
                "output": label,
            }
            messages.append(message)
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")