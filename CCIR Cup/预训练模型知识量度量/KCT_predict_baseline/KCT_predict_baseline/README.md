## 预训练基准模型知识量度量baseline说明

#### 1. 安装依赖 

##### 1.1 pip Installation

安装依赖包：numpy, pandas，torch，transformers

```
pip install -r requirements.txt
```

##### 1.2 下载BERT-BASE预训练模型：

pytorch版本下载链接: https://huggingface.co/bert-base-uncased/tree/main

#### 2. 执行脚本

```
python  KCT_BERT_BASE_Predict.py --model_path=bert-base-uncased/ --data_file=./KCT_train_public.txt --output=./bert_base_predictions.csv
```

