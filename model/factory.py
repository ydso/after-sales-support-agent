"""
为项目提供模型
使用方法，导入factory.oy文件，然后调用chat_model,embedding_model就可以使用聊天模型和嵌入模型了
"""
from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from utils.config_handler import rag_config
from dotenv import load_dotenv
#加载dotenv文件
load_dotenv()

#父类
class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass

#子类，实现rag聊天模型的初始化
class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return init_chat_model(model=rag_config["chat_model_name"])

#子类，实现rag嵌入模型的初始化
class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])

chat_model = ChatModelFactory().generate()
embedding_model = EmbeddingsFactory().generate()