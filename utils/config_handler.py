"""
yaml
k: v
使用方法：导入config_handler.py文件，然后引用最后那四个变量即可实现配置文件的导入
"""
import yaml
from utils.path_tool import get_abs_path


#获取rag的配置文件
def load_rag_config(config_path: str=get_abs_path('config/rag.yml'),encoding:str='utf-8'):
    with open(config_path, "r",encoding=encoding) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

#获取chroma的配置文件
def load_chroma_config(config_path: str=get_abs_path('config/chroma.yml'),encoding:str='utf-8'):
    with open(config_path, "r",encoding=encoding) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

#获取prompts的配置文件
def load_prompts_config(config_path: str=get_abs_path('config/prompts.yml'),encoding:str='utf-8'):
    with open(config_path, "r",encoding=encoding) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path('config/agent.yml'),encoding:str='utf-8'):
    with open(config_path, "r",encoding=encoding) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

#后续只需要引用这四个变量即可
rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()


if __name__ == '__main__':
    print(rag_config["embedding_model_name"])