# ==============================================
# 通用异步重试工具 - 指数退避装饰器
# 说明：仅用于协程函数；同步函数不适用（内部 await 会直接报错）
# ==============================================
import asyncio
from functools import wraps


def async_retry(max_times: int = 2, delay: float = 0.5):
    """
    函数功能与逻辑描述：
        构造一个异步重试装饰器工厂：包裹后的协程函数在抛出异常时会自动重试，直到成功或次数耗尽。
        退避策略为指数退避，第 i 次重试（i 从 0 起）等待 delay * 2^i 秒，即 0.5s / 1s / 2s ...
        最后一次失败后**抛出最后一次捕获到的原始异常**（不是包装异常），
        以便调用方按原始异常类型处理（如 mcpGateway 侧的 CONNECTION_EXC 判定）。
        注意本装饰器只捕获 Exception，不捕获 BaseException（KeyboardInterrupt 等会直接冒泡）；
        且**不对幂等性做任何判断**，是否可重试必须由调用方自行保证
        （写副作用场景应传 max_times=0 显式关闭，如 agentCore.parsers.json_parser 的做法）。
        待重试的协程由本函数返回的 decorator 负责包裹，装饰器本身不做任何调用。
    入参说明：
        max_times (int)：最大重试次数，**不含首次调用**，默认 2（即最多执行 3 次）。
        delay (float)：首次重试的等待秒数基数，后续按 2 的幂次增长，默认 0.5。
    返回值说明：
        function：装饰器函数 decorator(func)，其返回值为包裹后的异步函数 wrapper，
            保留原函数元信息（经 functools.wraps）。
    """
    def decorator(func):
        """
        函数功能与逻辑描述：
            装饰器本体：用 functools.wraps 保留原协程函数的 __name__ / __doc__ 等元信息，
            并把 wrapper 返回给调用方作为替代实现。
        入参说明：
            func：被装饰的异步函数（协程函数）。
        返回值说明：
            function：包裹后的异步函数 wrapper。
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            """
            函数功能与逻辑描述：
                实际执行体：循环执行 func 最多 max_times + 1 次。每次异常都暂存到 last_exception，
                在还有重试余量时按 delay * 2^i 退避等待后继续；循环结束仍失败则抛出
                最后一次的原始异常。成功时立即返回结果，不再进行任何后续尝试。
            入参说明：
                *args：原样透传给 func 的位置参数。
                **kwargs：原样透传给 func 的关键字参数。
            返回值说明：
                Any：func 成功执行时的返回值。
            异常说明：
                Exception：重试次数耗尽后，抛出最后一次捕获到的原始异常对象。
            """
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
