"""
为整个工程提供统一的绝对路径
使用方法：导入path.tool.py文件，调用get_abs_path方法获取绝对路径
"""
import os

def get_project_root()->str:
    """
    获取项目根目录
    :return: 字符串
    """
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取文件所在的文件夹的绝对路径
    current_dir = os.path.dirname(current_file)
    # 获取文件所在的文件夹的上一级文件夹的绝对路径
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path:str)->str:
    """
    传递相对路径，返回绝对路径
    :param relative_path:相对路径
    :return:绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path('data\\test.txt'))
