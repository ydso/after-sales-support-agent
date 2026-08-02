"""
工程项目中的日志处理
使用方法，直接导入logger_handler.py这个文件 ,然后使用最下面的logger获取日志管理器
"""
import logging
import os
from datetime import datetime
from unittest import main

from utils.path_tool import get_abs_path

#日志保存的根目录
LOG_ROOT = get_abs_path('logs')

#确保日志目录的存在
os.makedirs(LOG_ROOT, exist_ok=True)

#日志的格式配置  error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
        name:str = "agent",
        console_level:int = logging.INFO,
        file_level:int = logging.DEBUG,
        log_file = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #避免重复添加Handler
    if logger.handlers:
        return logger

    #控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    #配置文件输出
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')

    file_handler = logging.FileHandler(log_file,encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


#快捷获取日志管理器
logger = get_logger()


if __name__ == '__main__':
    logger.info('This is an info message')
    logger.error('This is an error message')
    logger.warning('This is a warning message')
    logger.debug('This is a debug message')