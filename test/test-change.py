import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = '/root/ft/internlm2-chat-7b'
# os.system(f'git clone https://code.openxlab.org.cn/qwertyuiopa12/Cyber-InternLM2-Chat-7B.git {base_path}')
# os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()


def chat(message, history):
    for response, history in model.stream_chat(tokenizer, message, history, top_p=0.7, temperature=1):
        yield response

def get_question(question_index):
    questions = [
    "如何理解王维的'诗中有画，画中有诗'的评价？",
    "杜甫的《春望》中'国破山河在'的下一句是什么？",
    "李清照的《如梦令》中'昨夜雨疏风骤'的下一句是什么？",
    "苏轼的《江城子·密州出猎》中'老夫聊发少年狂'的下一句是什么？",
    "王之涣的《登鹳雀楼》中'白日依山尽'的下一句是什么？",
    "李白的《静夜思》中'床前明月光'的下一句是什么？",
    "辛弃疾的《青玉案·元夕》中'东风夜放花千树'的下一句是什么？",
    "白居易的《赋得古原草送别》中'离离原上草'的下一句是什么？",
    "柳永的《雨霖铃》中'寒蝉凄切'的下一句是什么？",
    "陆游的《示儿》中'死去元知万事空'的下一句是什么？"
    ]
    return questions[question_index]
def quiz_battle(question_index, user_answer):
    question_index = int(question_index)
    # 假设有一个问题列表和答案列表
    questions = [
    "如何理解王维的'诗中有画，画中有诗'的评价？",
    "杜甫的《春望》中'国破山河在'的下一句是什么？",
    "李清照的《如梦令》中'昨夜雨疏风骤'的下一句是什么？",
    "苏轼的《江城子·密州出猎》中'老夫聊发少年狂'的下一句是什么？",
    "王之涣的《登鹳雀楼》中'白日依山尽'的下一句是什么？",
    "李白的《静夜思》中'床前明月光'的下一句是什么？",
    "辛弃疾的《青玉案·元夕》中'东风夜放花千树'的下一句是什么？",
    "白居易的《赋得古原草送别》中'离离原上草'的下一句是什么？",
    "柳永的《雨霖铃》中'寒蝉凄切'的下一句是什么？",
    "陆游的《示儿》中'死去元知万事空'的下一句是什么？"
    ]
    options = [
    ["A. 王维的诗作具有鲜明的画面感，其画作又富有诗意", "B. 王维的诗作和画作风格迥异", "C. 王维的诗作和画作都缺乏深度", "D. 王维的诗作和画作都过于抽象"],
    ["A. 城春草木深", "B. 感时花溅泪", "C. 恨别鸟惊心", "D. 烽火连三月"],
    ["A. 浓睡不消残酒", "B. 试问卷帘人", "C. 却道海棠依旧", "D. 知否？知否？应是绿肥红瘦"],
    ["A. 左牵黄，右擎苍", "B. 锦帽貂裘，千骑卷平冈", "C. 为报倾城随太守", "D. 亲射虎，看孙郎"],
    ["A. 黄河入海流", "B. 欲穷千里目", "C. 更上一层楼", "D. 白日依山尽"],
    ["A. 疑是地上霜", "B. 举头望明月", "C. 低头思故乡", "D. 床前明月光"],
    ["A. 更吹落、星如雨", "B. 宝马雕车香满路", "C. 凤箫声动，玉壶光转", "D. 一夜鱼龙舞"],
    ["A. 一岁一枯荣", "B. 野火烧不尽", "C. 春风吹又生", "D. 远芳侵古道"],
    ["A. 对长亭晚", "B. 骤雨初歇", "C. 都门帐饮无绪", "D. 留恋处、兰舟催发"],
    ["A. 但悲不见九州同", "B. 王师北定中原日", "C. 家祭无忘告乃翁", "D. 死去元知万事空"]
    ]
    correct_answers = [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A"
    ]
    # 获取问题和答案
    question = questions[question_index]
    correct_answer = correct_answers[question_index]
    answer_options = options[question_index]
    formatted_options = '\n'.join(answer_options)

    # 创建提示
    prompt = f"你好诗墨，下面是一道选择题和对应的选项，请你从给出的四个选项中选出一个正确答案，答案有且仅有一个，请注意，你的答案只能是A,B,C,D这四个字母中的一个:\n{question}\n{formatted_options}"
    # 使用模型生成答案
    model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, prompt, history=[])
    model_answer = response
    # model_answer = model.generate(tokenizer.encode(question, return_tensors='pt'), max_length=1, num_return_sequences=1)[0]
    # model_answer = tokenizer.decode(model_answer, skip_special_tokens=True)

    # 比较答案
    if model_answer == correct_answer:
        result = "大模型赢了"
    elif user_answer == correct_answer:
        result = "你赢了"
    else:
        result = "大模型和你都错了"

    #return {"question": question, "model_answer": model_answer, "result": result}
    return question, model_answer, result

examples = [
    ["请介绍一下文言文的概念"],
    ["简单介绍一下《桃花源记》"],
    ["请介绍王羲之的‘兰亭序’的典故"],
    ["请创作一首以落日为艺术表现的七言绝句"],
    ["请解释《琵琶行》中的'琵琶'象征意义"],
    ["你是由哪个团队开发的"],
    ["请创作一首以月下独酌为灵感的五言绝句"],
    ["请帮我科普一下唐朝诗人的有关知识"],
    ["我今天心情不好，学习有点累了，可以安慰我吗"],
    ["你的名字是什么"],
    ["你的英文名字是什么"]
]
iface = gr.ChatInterface(
  chat,
  title="PoeticCalligraphy",
  description="""
我是古文诗词领域助手诗墨，欢迎向我提问.  
                 """,
  examples=examples
  )

quiz_iface = gr.Interface(
    fn=quiz_battle,
    inputs=[
        gr.Slider(minimum=0, maximum=9, step=1,  label="选择题号"),
        gr.Radio(choices=["A", "B", "C", "D"], label="你的答案")
    ],
    outputs=[
        gr.Textbox(label="问题"),
        gr.Label(label="大模型答案"),
        gr.Textbox(label="结果")
    ],
    title = "古文诗词选择题对战",
    description ="挑战大模型，看看谁更懂古文诗词！"
)

# 组合两个界面
quiz_iface.inputs[0].change(fn=get_question, inputs=quiz_iface.inputs[0], outputs=quiz_iface.outputs[0])
demo = gr.TabbedInterface([iface, quiz_iface], ["聊天", "选择题对战"])
demo.queue(1).launch()
