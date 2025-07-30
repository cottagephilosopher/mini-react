from setuptools import setup, find_packages

setup(
    name="mini-react",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.23.0",   # HTTP客户端，用于调用API
        "loguru",          # 日志处理
        "python-dotenv",   # 环境变量管理
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],  # OpenAI官方客户端库
        "all": ["openai>=1.0.0"],  # 所有可选依赖
    },
    author="alex",
    author_email="thisgame@foxmail.com",
    description="轻量级 ReAct 智能体框架实现",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cottagephilosopher/mini-react",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)