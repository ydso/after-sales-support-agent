"""
rag总结服务：用户提问，搜索参考资料，然后结合提问和参考资料，提交给模型，让模型总结回复
"""
from langchain_core.output_parsers import StrOutputParser
from rag.retrieval import HybridRetriever, QueryRewriter, RetrievalConfig
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from utils.config_handler import chroma_config


class RagSummarizeService(object):
    #初始化向量存储，检索器，提示词文本，提示词模板，模型，以及rag执行的链
    def __init__(self):
        self.vector_store = VectorStoreService()
        retrieval_config = RetrievalConfig.from_mapping(chroma_config)
        # 混合检索器保持 invoke(query) 接口，避免改变上层工具调用逻辑。
        self.retriever = HybridRetriever(
            vector_store=self.vector_store.vector_store,
            query_rewriter=QueryRewriter(
                model=chat_model,
                max_rewrites=retrieval_config.max_rewrites,
            ),
            config=retrieval_config,
        )
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    #构建链
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    #用户提问检索器
    def retriever_docs(self,query:str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self,query:str) -> str:
        #获取检索到的参考文档
        context_docs = self.retriever_docs(query)

        #构建参考文档
        context = ""
        counter =0
        for doc in context_docs:
            counter +=1
            context += f"【参考资料{counter}】，内容：\n{doc.page_content} | 参考元数据{doc.metadata}\n"


        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

if __name__ == "__main__":
    rag_service = RagSummarizeService()

    for chunk in rag_service.rag_summarize(
        "APP登录失败，无法绑定机器人怎么办"
    ):
        print(chunk, end="", flush=True)
