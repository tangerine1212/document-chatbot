# 语义划分RAG增强的可信问答系统

该项目是课程自然语言处理的期末大作业，在大模型微调上主要借鉴了[TruthReader](https://github.com/HITsz-TMG/TruthReader-document-assistanthttps://)，并实现了训练代码，在此基础上作出了一些改进。

## 环境

```
pip install -r requirements.txt
```

## Web

通过下列链接下载相应权重，放在主目录下并命名为"models"
链接: https://pan.baidu.com/s/11OYX48BaaZcjb-ZSHHUZJg?pwd=x5c6 提取码: x5c6 

运行

```
python app.py --config_path yamls/chatbot.yaml
```

## 训练

下载对应预训练大模型（如Qwen2.5-14B-Instruct），并修改对应路径至yamls/train.yaml文件中

运行

```
python train.py --config_path yamls/train.yaml
```

## RAG

RAG相关代码及可视化在文件夹RAG中
