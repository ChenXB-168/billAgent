import asyncio
from functools import wraps

def async_retry(max_times: int = 2, delay: float = 0.5):
    """
    通用异步重试装饰器
    :param max_times: 最大重试次数（不含首次）
    :param delay: 首次重试等待秒数，指数退避
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(max_times + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if i < max_times:
                        wait_time = delay * (2 ** i)
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator