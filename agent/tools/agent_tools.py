"""
定义agent要使用的工具

工具有：rag_summarize，get_user_id，get_current_month，fetch_external_data，fill_context_for_report，get_current_date
"""
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import os
from utils.path_tool import get_abs_path
from utils.config_handler import agent_config

rag = RagSummarizeService()

external_data={}

@tool
def rag_summarize(query:str) -> str:
    """
    从向量数据库中检索资料
    :param query:
    :return:
    """
    return rag.rag_summarize(query)


def mask_user_id(user_id: str) -> str:
    """对日志中的用户 ID 进行脱敏。"""

    if len(user_id) <= 2:
        return "*" * len(user_id)

    return f"{user_id[:2]}{'*' * (len(user_id) - 2)}"


@tool
def get_user_id() -> str:
    """
    获取当前用户的唯一 ID。

    当前项目尚未接入登录系统，因此开发和演示阶段从
    DEMO_USER_ID 环境变量中读取固定的演示用户 ID。

    当用户需要查询个人使用记录或生成月度报告，
    且尚未获得用户 ID 时，调用此工具。

    Returns:
        数字字符串格式的用户 ID，例如 1001。
    """

    user_id = os.getenv("DEMO_USER_ID", "").strip()

    if not user_id:
        logger.error(
            "[获取用户ID失败] 未配置 DEMO_USER_ID"
        )

        return (
            "获取用户ID失败：当前系统尚未配置演示用户，"
            "暂时无法查询个人使用报告。"
        )

    if not user_id.isdigit():
        logger.error(
            "[获取用户ID失败] DEMO_USER_ID格式错误 value=%s",
            user_id,
        )

        return (
            "获取用户ID失败：DEMO_USER_ID 必须是数字字符串。"
        )

    logger.info(
        "[获取用户ID成功] 当前使用演示用户 user_id=%s",
        mask_user_id(user_id),
    )

    return user_id


@tool
def get_current_month() -> str:
    """
    获取系统当前月份。

    当用户生成报告但没有指定月份，或者用户说“本月”
    “这个月”时，调用此工具。

    Returns:
        YYYY-MM 格式的月份，例如 2026-07。
    """

    timezone_name = "Asia/Shanghai"

    try:
        current_month = datetime.now(
            ZoneInfo(timezone_name)
        ).strftime("%Y-%m")

    except ZoneInfoNotFoundError:
        logger.exception(
            "[获取当前月份失败] 无法识别时区 timezone=%s",
            timezone_name,
        )

        return "获取当前月份失败：系统时区配置无效。"

    logger.info(
        "[获取当前月份成功] timezone=%s month=%s",
        timezone_name,
        current_month,
    )

    return current_month

def generate_external_data():
    """
    {
        "user_id":{
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            ....
        },
        "user_id":{
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            ....
        },
        "user_id":{
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            "month":{"特征":xxx,"效率":xxx,....}
            ....
        }
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"',"")
                feature: str = arr[1].replace('"',"")
                efficiency: str = arr[2].replace('"',"")
                consumables: str = arr[3].replace('"',"")
                comparison: str = arr[4].replace('"',"")
                time: str = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }

@tool
def fetch_external_data(
    user_id: str,
    month: str,
) -> str:
    """
    查询指定用户在指定月份的机器人使用记录。

    Args:
        user_id: 数字字符串格式的用户 ID。
        month: YYYY-MM 格式的月份。

    Returns:
        结构化的使用记录字符串。
    """

    generate_external_data()

    try:
        return external_data[user_id][month]

    except KeyError:
        logger.error(f"[generate_external_data]未能检索到用户：{user_id}在{month}的使用记录")
        return ""

@tool
def fill_context_for_report():
    """
    无入参，无返回值，调用后触发中间件自动为报告生成的场景注入上下文信息,为后续提示词切换提供上下文信息
    :return:
    """
    return "fill_context_for_report已调用"

@tool
def get_current_date() -> str:
    """
    获取当前日期。

    当搜索最新天气、新闻、产品和实时信息时，
    调用此工具确定准确日期。
    """
    return datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y-%m-%d")
