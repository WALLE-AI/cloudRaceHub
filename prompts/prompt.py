INTENT_LIST = '''
信用卡办理
贷款业务
车险办理
银行卡办理
'''

INTENT_CHAT_PROMPT = '''你是一个智能助手，你能够根据当前用户的输入和历史对话信息，准确的识别出当前用户输入的意图，请根据如下意图类别进行回复：

用户输入：{content}
如果用户输入与当前任务无关，无需进行回复，请按如下格式进行输出：
{
    intent:xxxx
}

'''

TEXT_CLASSFICATION_PROMPT_ZH = '''
你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型

'''