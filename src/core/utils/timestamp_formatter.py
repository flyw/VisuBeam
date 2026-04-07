"""
时间戳格式化工具，用于在日志中显示毫秒级时间戳
格式为: YYYY-MM-dd HH:mm:ss.SSS
"""
import logging
from datetime import datetime


class MillisecondFormatter(logging.Formatter):
    """
    自定义日志格式化器，包含毫秒级时间戳
    格式为: YYYY-MM-dd HH:mm:ss.SSS
    """
    def formatTime(self, record, datefmt=None):
        """
        格式化时间戳为指定格式
        """
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds (3 digits)


def setup_logging_with_milliseconds(level=logging.INFO, format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """
    设置包含毫秒级时间戳的日志记录
    
    Args:
        level: 日志级别
        format_string: 日志格式字符串
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 移除已有的处理器，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    formatter = MillisecondFormatter(format_string)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger