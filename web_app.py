"""网页启动入口：先设置数值库兼容环境，再加载会话服务。"""
import os

# 与CLI使用相同兼容策略，避免Windows混合加载MKL/PyTorch发生OpenMP冲突。
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

from modules.web.api import create_app

app = create_app()


if __name__ == '__main__':
    import uvicorn

    # 直接复用已创建的应用，以单进程运行，避免重复初始化和多个分析队列争用模型。
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)
