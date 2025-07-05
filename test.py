from src.chatbot import DocumentQASystem
import yaml
from pathlib import Path

def main(config):
    chatbot = DocumentQASystem(config)
    questions = ["大语言模型的定义是什么？", "GPT-3的参数规模是多少？", "2025年国际消费电子展高通展台的人形机器人叫什么名字？", "N-gram模型的哪些局限性促使研究者转向神经网络语言模型？", "大语言模型的风险与挑战有哪些方面？"]
    test_question = ["GPT-3的参数规模是多少？"]
    # questions = test_question
    file_paths = ["大语言模型.docx"]
    chatbot.build_vector_database(file_paths)
    results = []
    content = ""
    for question in questions:
        results.append(chatbot.answer_question(question))
    for question, result in zip(questions, results):
        content += "- "*50+'\nquestion:\n'+question+'\nanswer:\n'+result["prediction"]+'\n'
    with open("./output/results.txt", "w") as file:
        file.write(content)
        
if __name__ == "__main__":
    config_path = 'yamls/chatbot.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)