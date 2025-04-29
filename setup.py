from setuptools import setup, find_packages

setup(
    name="miniReAct",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=0.1.1",  # 用于处理大语言模型请求
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
    url="https://github.com/longxtx/mini-react",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
) 