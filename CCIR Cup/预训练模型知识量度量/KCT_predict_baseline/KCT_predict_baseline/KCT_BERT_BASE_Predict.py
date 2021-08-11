"""Pretrained BERT-BASE model as baseline version 1.0.
To use this script, use command python KCT_BERT_BASE_Predict.py <预训练模型> <训练集或者测试集文件> <预测结果>
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser("Pretrained BERT-BASE model as baseline.")
    parser.add_argument('--model_path', metavar='bert-base-uncased/', help='Input bert base model path.')   # BERT-BASE模型路径
    parser.add_argument('--data_file', metavar='KCT_train_public.txt', help='Input data file.')    # 训练集或者测试集文件
    parser.add_argument('--output', metavar='bert_base_predictions.csv', help='output data file.')  # 输出路径
    parser.add_argument('--max_token_length', default=5, type=int, help='max length of predict tokens')  # 最大预测token长度

    if len(sys.argv) != 4:
        print('argument has error,' + str(len(sys.argv)) + ' not equal 4')
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_model(model_path):
    try:
        bert_model = BertForMaskedLM.from_pretrained(model_path)
        bert_model.eval()
    except Exception as e:
        print(f"model loading error： {e}")
        bert_model = None
    return bert_model


def get_tokenizer(model_path):
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"tokenizer loading error： {e}")
        bert_tokenizer = None
    return bert_tokenizer


def maskRemove(masked_sentences):
    # 构造MASK前后句
    try:
        start_string, end_string = masked_sentences.strip().split("[MASK]")
    except Exception as e:
        print(masked_sentences, e)
        start_string, end_string = "", ""
    return [start_string, end_string]


def get_batch_data(src_data, batch_limit):
    # 保存qid信息
    qid_list = [item["qid"] for item in src_data]

    # 依次取出sentences
    maskSents = [maskRemove(item["query"]) for item in src_data]

    # 依据batch size进行数据转换
    sentReshaped = [maskSents[i:i + batch_limit] for i in range(0, len(maskSents), batch_limit)]

    if len(qid_list) == len(src_data):
        print(f"----》 data reshaped，total：{len(src_data)}, batch count：{len(sentReshaped)}")
    else:
        print("读取数量不符")
        qid_list, sentReshaped = [], []
    return qid_list, sentReshaped


def get_top_answers(pred_mask_tokens, topK=5):
    """
    :param pred_mask_tokens: 预测结果
    :param topK: 输出预测结果数量
    :return: 对预测结果依据weight排序，取前topK为预测结果输出
    """
    res = []
    for k_pred in pred_mask_tokens:
        k_pred_res = []
        for pred in k_pred:
            if len(pred) == 1:
                k_pred_res.append(pred[0])
            else:
                k_pred_res.append((tokenizer.convert_tokens_to_string([item[0] for item in pred]), np.mean([item[1] for item in pred])))
        res.append(sorted(k_pred_res, key=lambda x: x[1], reverse=True)[:topK])
    return res


def bert_inference(inputs, topk=3):
    with torch.no_grad():
        inputs = tokenizer(inputs, padding=True, return_tensors='pt')
        inputs = {k: v for k, v in inputs.items()}
        mask_positions = (inputs['input_ids'] == tokenizer.convert_tokens_to_ids('[MASK]'))
        # [B, L, D]
        predictions = model(**inputs)[0]
        # [N, D]
        predictions = predictions[mask_positions]
        predictions = predictions.softmax(dim=-1)
        num_masked_tokens = predictions.size(0)
        # [N, D]
        sorted_preds, sorted_idx = predictions.sort(dim=-1, descending=True)

        predicted_tokens = []
        for k in range(topk):  # topk
            predicted_index = sorted_idx[:, k]
            predicted_score = sorted_preds[:, k]
            predicted_token = [(tokenizer.convert_ids_to_tokens(x), score)
                               for (x, score)
                               in zip(predicted_index.tolist(), predicted_score.tolist())]

            predicted_tokens.append(predicted_token)

    return predicted_tokens


def main(src_data, max_token_length, topK=5, batch_size=10):
    # step 1: 有效信息提取转换，将query句子转换为大小为10的batch
    qid_list, sentItems_batch = get_batch_data(src_data, batch_size)

    pred_res_items = []     # 最终输出结果
    if not (qid_list or sentItems_batch):
        return pred_res_items

    predTokens_list = []    # tokens长度为1~5的所有预测结果
    for i in range(max_token_length):   # 依据 MASK 的最多为5进行迭代
        predicted_tokens_Limited = []   # [MASK] 限定长度的预测结果

        # step 2: 每个batch进行bert-base模型预测
        for examples in sentItems_batch:
            # 构造带有固定 [MASK] 长度的query
            examples = [item[0] + "".join(['[MASK] '] * (i + 1)) + item[1] for item in examples]
            predBatchTokens = bert_inference(examples, topk=topK)
            predicted_tokens_Limited.append(predBatchTokens)

        # step 3:预测结果batch还原为与qid的对应数组
        start_index = 0     # 起始指针
        for j in range(len(qid_list)):
            # qid所在batch索引
            batch_index = j // batch_size
            if j % batch_size == 0:
                # 切片起始位置归零
                start_index = 0

            batch_pred_tokens = predicted_tokens_Limited[batch_index]
            sent_pred_tokens = [top_tokens[start_index: start_index + i + 1] for top_tokens in batch_pred_tokens]

            if i == 0:
                predTokens_list.append(sent_pred_tokens)    # 首轮需要创建每个qid对应的答案列表
            else:
                predTokens_list[j] += sent_pred_tokens
            start_index += (i + 1)
    # step 4: 预测结果取前5
    top_answers = get_top_answers(predTokens_list, topK)

    if len(top_answers) == len(qid_list):
        for i in range(len(qid_list)):
            pred_res_items.append({"id": qid_list[i], "ret": json.dumps([tup[0] for tup in top_answers[i]])})
    else:
        print("预测结果与输出数量不匹配")
    return pred_res_items


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    KCT_file_path = args.data_file
    max_token_length = args.max_token_length
    output_path = args.output

    print(f"loading model...")
    model = get_model(model_path)   # 加载预训练模型

    print(f"loading tokenizer...")
    tokenizer = get_tokenizer(model_path)   # 加载tokenizer
    print(f"----->>>model and tokenizer loaded...")

    print(f"start Bert-BASE model task...")

    with open(KCT_file_path, "r") as f:
        read_lines = [json.loads(line.strip()) for line in f.readlines()]

    topK = 5
    batch_size = 10
    res = main(read_lines, max_token_length, topK, batch_size)
    if res:
        df = pd.DataFrame(res)
        df.to_csv(output_path, index=False)
        print(df.head())
        print("---->>>> task complete...")
    else:
        print("任务失败...")