"""
搜索示例程序，展示如何使用miniReAct框架创建一个搜索和问答智能体
"""
import os
import sys
import logging

# 添加上级目录到模块搜索路径，以便导入miniReAct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import miniReact as mr

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 模拟数据库，存储一些文档
documents = {
    "人工智能": """
    人工智能（Artificial Intelligence，简称AI）是指由人制造出来的机器所表现出来的智能。
    通常人工智能是指通过普通计算机程序来呈现人类智能的技术。
    也指出研究各种智能机器的设计原理与实现方法，使机器能在特定环境中感知、推理与行动。
    
    人工智能的研究领域包括机器学习、计算机视觉、自然语言处理等。
    深度学习是当前人工智能发展的重要技术方向。
    """,
    
    "机器学习": """
    机器学习是人工智能的一个分支，是一种能够让计算机在没有被明确编程的情况下学习的方法。
    机器学习算法基于样本数据（训练数据）构建数学模型，以便在没有明确编程的情况下进行预测或决策。
    
    机器学习主要分为监督学习、无监督学习和强化学习三种类型。
    常见的机器学习算法包括决策树、支持向量机、神经网络等。
    """,
    
    "深度学习": """
    深度学习是机器学习的一个分支，它基于人工神经网络的架构进行学习。
    深度学习通过组合多层简单但非线性的模块，可以学习数据的多层次表示。
    
    卷积神经网络（CNN）和循环神经网络（RNN）是深度学习中最常用的两种网络结构。
    深度学习在图像识别、语音识别和自然语言处理等领域取得了重大突破。
    """,
    
    "自然语言处理": """
    自然语言处理（Natural Language Processing，简称NLP）是人工智能的一个子领域，
    致力于使计算机能够理解、解释和生成人类语言。
    
    NLP的应用包括机器翻译、情感分析、文本摘要、问答系统等。
    大型语言模型如GPT和BERT是当前NLP领域的重要进展。
    """
}

# 定义搜索工具
def search(query: str) -> str:
    """
    搜索数据库中与查询相关的文档
    
    参数:
        query: 搜索查询字符串
        
    返回:
        搜索结果字符串
    """
    # 简单的关键词匹配
    results = []
    for title, content in documents.items():
        if query.lower() in title.lower() or query.lower() in content.lower():
            results.append(f"- {title}: {content[:100]}...")
    
    if results:
        return "找到以下相关文档:\n" + "\n".join(results)
    else:
        return f"没有找到与'{query}'相关的文档。"

# 定义阅读工具
def read_document(title: str) -> str:
    """
    阅读指定标题的文档
    
    参数:
        title: 文档标题
        
    返回:
        文档内容
    """
    # 尝试精确匹配
    if title in documents:
        return f"文档 '{title}':\n{documents[title]}"
    
    # 尝试模糊匹配
    for doc_title in documents:
        if title.lower() in doc_title.lower() or doc_title.lower() in title.lower():
            return f"找到相似标题文档 '{doc_title}':\n{documents[doc_title]}"
    
    return f"找不到标题为'{title}'的文档。"

# 定义任务签名
qa_signature = mr.Signature(
    {"question": mr.InputField(desc="用户的问题")},
    {"answer": mr.OutputField(desc="回答"), "sources": mr.OutputField(desc="信息来源")},
    instructions="你是一个问答智能体，能够回答用户关于人工智能的问题。使用提供的工具搜索相关信息，并给出准确的回答。"
)

def main():
    # 创建工具列表
    tools = [search, read_document]
    from llm_hub import MultiLLMHub
    lm = MultiLLMHub().setup_azure_openai()
    # 创建ReAct智能体
    agent = mr.ReAct(qa_signature, tools,3,lm=lm)
    
    print("欢迎使用AI知识问答智能体！")
    print("输入你的问题，或输入 'exit' 退出")
    print("例如: '什么是深度学习?' 或 '人工智能和机器学习有什么区别?'")
    
    while True:
        user_input = input("\n问题> ")
        if user_input.lower() in ('exit', 'quit', 'q'):
            break
        
        # 使用智能体处理问题
        try:
            print("\n正在思考...")
            result = agent(question=user_input)
            
            print(f"\n回答: {result.answer}")
            print(f"\n信息来源: {result.sources}")
            
            # 可选：打印轨迹信息，用于调试
            if '--debug' in sys.argv:
                print("\n轨迹详情:")
                for key, value in result.trajectory.items():
                    if key.startswith('thought'):
                        print(f"\n思考: {value}")
                    elif key.startswith('tool_name'):
                        idx = key.split('_')[-1]
                        print(f"工具: {value}")
                        print(f"参数: {result.trajectory.get(f'tool_args_{idx}', {})}")
                        print(f"观察: {result.trajectory.get(f'observation_{idx}', '')}")
                
        except Exception as e:
            print(f"处理问题时出错: {e}")
            
if __name__ == "__main__":
    # 设置环境变量以指定要使用的模型
    
    # 如果想查看执行过程中的思考和工具调用
    mr.enable_debug()
    # 启动主程序
    main() 