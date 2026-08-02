"""
提示词加载
"""
from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    #第一步读取配置项中的system_prompt_path路径
    try:
        system_prompts_path = get_abs_path(prompts_config['system_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在prompts.yaml配置项中没有system_prompt_path配置项")
        raise e

    #第二步,读取获取到的文件路径中的内容
    try:
        with open(system_prompts_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_prompts():
    #第一步读取配置项中的rag_summarize_prompt_path路径
    try:
        rag_prompts_path = get_abs_path(prompts_config['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在prompts.yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    #第二步,读取获取到的文件路径中的内容
    try:
        with open(rag_prompts_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e

def load_report_prompts():
    #第一步读取配置项中的report_prompt_path路径
    try:
        report_prompts_path = get_abs_path(prompts_config['report_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在prompts.yaml配置项中没有report_prompt_path配置项")
        raise e

    #第二步,读取获取到的文件路径中的内容
    try:
        with open(report_prompts_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e

if __name__ == '__main__':
    print(load_system_prompts())
