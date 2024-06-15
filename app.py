import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = './internlm2-chat-7b'
os.system(f'git clone https://code.openxlab.org.cn/qwertyuiopa12/shimo.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response
        
examples = [
    ["请介绍一下文言文的概念"],
    ["简单介绍一下《桃花源记》"],
    ["请介绍王羲之的‘兰亭序’的典故"],
    ["请创作一首以落日为艺术表现的七言绝句"],
    ["请解释《琵琶行》中的'琵琶'象征意义"],
    ["你是由哪个团队开发的"],
    ["你的英文名字是什么"],
    ["请创作一首以月下独酌为灵感的五言绝句"],
    ["我今天心情不好，学习有点累了，可以安慰我吗"],
    ["你的名字是什么"],
    ["请帮我科普一下唐朝诗人的有关知识"]
]

gr.ChatInterface(chat,
                 title="PoeticCalligraphy",
                description="""
我是古文诗词领域助手诗墨，由水滴石穿团队开发，欢迎和我一起探索绚丽的古文诗词世界！ 
                 """,
                 examples=examples 
                 ).queue(1).launch()
