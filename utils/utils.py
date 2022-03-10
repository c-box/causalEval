from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    GPT2LMHeadModel, GPT2Tokenizer, \
    RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, BertForMaskedLM, \
    BartForConditionalGeneration, BartTokenizer, XLNetTokenizer, T5Tokenizer
import torch
import json
from utils.constant import CUDA_DEVICE, RELATION_FILES
import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable
import seaborn as sns
import os
import pandas as pd
from matplotlib import rcParams
import heapq

# 各个模型的存储位置
MODEL_PATH = {
    # 'bert-base-uncased': '/share/model/bert/uncased_L-12_H-768_A-12',
    # 'bert-base-cased': '/home/caoboxi/saved_models/bert-base-cased',
    # 'bert-large-uncased': '/share/model/bert/uncased_L-24_H-1024_A-16',
    # 'bert-large-cased': '/home/caoboxi/saved_models/bert-large-cased',
    # 'bert-large-cased-wwm': '/home/caoboxi/saved_models/bert-large-cased-whole-word-masking',

    # 'gpt2': '/home/caoboxi/saved_models/gpt2',
    # "gpt2-medium": "/home/caoboxi/saved_models/gpt2-medium",
    # 'gpt2-large': '/home/caoboxi/saved_models/gpt2-large',
    # 'gpt2-xl': '/home/caoboxi/saved_models/gpt2-xl',

    # 'roberta-base': '/home/caoboxi/saved_models/roberta-base',
    # 'roberta-large': '/home/caoboxi/saved_models/roberta-large',

    'bart-large': 'facebook/bart-large',
    # 'bart-base': '/home/caoboxi/saved_models/bart-base'
}


def build_model(model_name):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
    else:
        raise RuntimeError('model not exsit')
    if model_prefix(model_name) == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path, force_bos_token_to_be_generated=True)
    elif model_prefix(model_name) == "gpt2":
        tokenizer, model = build_gpt_model(model_name)
    elif model_prefix(model_name) == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
    elif model_prefix(model_name) == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    return tokenizer, model


def build_gpt_model(model_name):
    if model_name in MODEL_PATH:
        model_type = MODEL_PATH[model_name]
    else:
        raise RuntimeError('model not exsit')
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = GPT2LMHeadModel.from_pretrained(model_type, return_dict=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def batch_text(input_texts, batch_size=32, obj_tokens=None):
    if obj_tokens is None:
        batch_input_text = []
        single_batch = []
        for text in input_texts:
            single_batch.append(text)
            if len(single_batch) == batch_size:
                batch_input_text.append(single_batch)
                single_batch = []
        if len(single_batch) > 0:
            batch_input_text.append(single_batch)
        return batch_input_text
    else:
        assert len(input_texts) == len(obj_tokens)
        batch_input_text = []
        batch_obj_tokens = []
        single_batch = []
        single_obj_batch = []
        for text, obj in zip(input_texts, obj_tokens):
            single_batch.append(text)
            single_obj_batch.append(obj)
            if len(single_batch) == batch_size:
                batch_input_text.append(single_batch)
                batch_obj_tokens.append(single_obj_batch)
                single_batch = []
                single_obj_batch = []
        if len(single_batch) > 0:
            batch_input_text.append(single_batch)
            batch_obj_tokens.append(single_obj_batch)
        return batch_input_text, batch_obj_tokens


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
        f.close()
    return data


def store_file(filename, data):
    with open(filename, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
    f.close()


def load_json_dic(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def store_json_dic(filename, dic):
    with open(filename, 'w') as f:
        json.dump(dic, f)


def load_roberta_vocab():
    return load_json_dic("data/roberta_data/roberta_vocab.json")


def model_prefix(model_name):
    return model_name.split("-")[0]


def filter_samples_by_vocab(samples, vocab):
    filter_samples = []
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj in vocab:
            filter_samples.append(sample)
    return filter_samples, len(samples), len(filter_samples)


def get_relations(file_path='data/relations_with_trigger.jsonl'):
    original_relations = load_file(file_path)
    return original_relations


def score_to_result(score, topk, tokenizer, obj_token=None, rank_k=10000, roberta_vocab2idx=None):
    score = torch.softmax(score, dim=-1)
    predicted_prob, predicted_index = torch.topk(score, topk)
    predicted_prob = predicted_prob.detach().cpu().numpy()
    predicted_index = predicted_index.cpu().numpy().tolist()
    if "roberta" in tokenizer.name_or_path:
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(tokenizer.decode(index).strip())
    elif 'bert' in tokenizer.name_or_path:
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index)
    elif 'gpt' in tokenizer.name_or_path:
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(tokenizer.decode(index))
    else:
        raise RuntimeError('model not defined')
    if obj_token is None:
        return predicted_tokens, predicted_prob
    else:
        if "roberta" in tokenizer.name_or_path:
            if roberta_vocab2idx is None:
                raise RuntimeError("need to be fix")
            obj_index = roberta_vocab2idx[obj_token]
            obj_prob = score[obj_index].item()
        else:
            obj_index = tokenizer.convert_tokens_to_ids(obj_token)
            obj_prob = score[obj_index].item()

        rank_prob, rank_index = torch.topk(score, rank_k)
        rank_index = rank_index.cpu().numpy().tolist()
        if obj_index not in rank_index:
            obj_rank = rank_k
            mrr = 0
        else:
            obj_rank = rank_index.index(obj_index) + 1
            mrr = 1 / obj_rank
        return predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr


def get_pair(sample, return_id=False):
    while "sub_label" not in sample:
        try:
            sample = sample['sample']
        except:
            print(sample)
            exit(0)
    sub = sample['sub_label']
    obj = sample['obj_label']
    sub_id = sample["sub_uri"]
    if return_id:
        return sub, obj, sub_id
    else:
        return sub, obj


def mean_round(num, num_len, r=2):
    return round(num * 100 / num_len, r)


def divide_samples_by_ans(samples):
    true_samples = []
    false_samples = []
    for sample in samples:
        if sample['predict_ans'] is True:
            true_samples.append(sample)
        else:
            false_samples.append(sample)
    return true_samples, false_samples


def box_plot(ax, data, labels=None):
    ax.boxplot(data, labels=labels)
    plt.show()


def get_relation_args(args):
    infos = RELATION_FILES[args.relation_type]
    args.relation_file = infos['relation_file']
    args.sample_dir = infos['sample_dir']
    args.sample_file_type = infos["sample_file_type"]
    return args


def get_bert_vocab(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    return vocab


def set_seed(seed_num=1023):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True


def get_table_stat(table: PrettyTable, return_cols=False, return_mean=False):
    rows = table._rows
    mean_row = []
    median_row = []
    up_quantile_row = []
    down_quantile_row = []
    std_row = []

    cols = []

    if len(rows) == 0:
        return table

    for j in range(len(rows[0])):
        cols.append([row[j] for row in rows])

    for col in cols:
        if type(col[0]) == str:
            mean_row.append('mean')
            median_row.append('median')
            std_row.append('std')
            up_quantile_row.append('up_quantile')
            down_quantile_row.append('down_quantile')
        else:
            mean = round(float(np.mean(col)), 2)
            mean_row.append(mean)
            median = round(float(np.median(col)), 2)
            median_row.append(median)
            std = round(float(np.std(col)), 2)
            std_row.append(std)
            up_quantile = round(float(np.quantile(col, 0.25)), 2)
            up_quantile_row.append(up_quantile)
            down_quantile = round(float(np.quantile(col, 0.75)), 2)
            down_quantile_row.append(down_quantile)

    table.add_row(mean_row)
    table.add_row(up_quantile_row)
    table.add_row(median_row)
    table.add_row(down_quantile_row)
    table.add_row(std_row)
    if return_cols is False and return_mean is False:
        return table
    if return_cols is True:
        return table, cols
    if return_mean is True:
        return table, mean_row, std_row


def draw_heat_map(data, row_labels, col_labels,
                  pic_dir='pics/paper_pic/head_or_tail', pic_name='all_samples'):
    plt.figure(figsize=(8, 2))
    sns.set_theme()
    ax = sns.heatmap(data=data,
                     center=0,
                     annot=True, fmt='.2f',
                     xticklabels=row_labels,
                     yticklabels=col_labels)
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig('{}/{}.eps'.format(pic_dir, pic_name), format='eps')
    plt.show()


def draw_box_plot(corrs, pic_name, pic_dir, ylim=None, hor=True):
    data = {"prompt": [], "corr": []}
    for prompt in corrs:
        for corr in corrs[prompt]:
            data["prompt"].append(prompt)
            data["corr"].append(corr)

    pd_data = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    if hor is True:
        flatui = ["#d6ecfa"]
        ax = sns.boxplot(
            x="corr", y="prompt",
            data=pd_data, orient='h', width=.6,
            boxprops={'color': '#404040',
                      'facecolor': '#d6ecfa'
                      }
        )
    else:
        ax = sns.boxplot(
            x="prompt", y="corr",
            data=pd_data, width=.3,
            palette="Set2"
        )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    for line in ax.get_lines():
        line.set_color("#404040")

    set_plt()
    if ylim is not None:
        ax.set(ylim=ylim)
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    fig = ax.get_figure()
    fig.savefig('{}/{}.eps'.format(pic_dir, pic_name), format='eps')


def draw_corr_scatter(data, pic_name, pic_dir, prompt="T_{man}"):
    pd_data = pd.DataFrame(data)
    pd_data = pd_data[pd_data["prompts"] == prompt]
    print(pd_data)
    ax = sns.regplot(x="kl", y="precision", data=pd_data)
    print("mean: {}".format(pd_data["kl"].mean()))


def draw_context_box_plot(true_p, false_p, obj_true_p, obj_false_p):
    data = {"prediction": [], "context": [], "precision": []}
    for p in true_p:
        data["precision"].append(p)
        data["prediction"].append("right")
        data["context"].append("mask obj")
    for p in false_p:
        data["precision"].append(p)
        data["prediction"].append("false")
        data["context"].append("mask obj")

    for p in obj_true_p:
        data["precision"].append(p)
        data["prediction"].append("right")
        data["context"].append("obj only")
    for p in obj_false_p:
        data["precision"].append(p)
        data["prediction"].append("false")
        data["context"].append("obj only")

    pd_data = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x="prediction", y="precision", hue="context",
        data=pd_data, palette="Set3", width=.3
    )


def load_wikidata(data_name, relation_id):
    return load_json_dic("data/wikidata/{}/{}".format(data_name, relation_id))


def delete_overlap_by_lower(samples):
    temp_samples = []
    for sample in samples:
        sub, obj = get_pair(sample)
        sub = sub.lower()
        obj = obj.lower()
        if obj in sub:
            continue
        else:
            temp_samples.append(sample)
    return temp_samples


def print_predictions(sentence, preds_probs):
    k = min(len(preds_probs),10)
    # print(f"Top {k} predictions")
    print("-------------------------")
    print(f"Rank\tProb\tPred")
    print("-------------------------")
    for i in range(k):
        preds_prob = preds_probs[i]
        print(f"{i+1}\t{round(preds_prob[1],3)}\t{preds_prob[0]}")

    print("-------------------------")
    # print("\n")
    print("Top1 prediction sentence:")
    print(f"\"{sentence.replace('[Y]',preds_probs[0][0])}\"")


def set_plt():
    config = {
        "font.family": 'serif',
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)


def count_distinct_obj(samples):
    objs = set()
    for sample in samples:
        sub, obj = get_pair(sample)
        if obj not in objs:
            objs.add(obj)
    return len(objs)


# 只要完整词
def get_whole_vocab2idx(tokenizer):
    # model_type = model_prefix(model_name)
    vocab = tokenizer.get_vocab()
    vocab2idx = {}
    if isinstance(tokenizer, BertTokenizer):
        for token in vocab:
            word = tokenizer.decode([vocab[token]]).strip()
            if "#" not in token:
                vocab2idx[word] = vocab[token]
    elif isinstance(tokenizer, RobertaTokenizer)\
            or isinstance(tokenizer, BartTokenizer)\
            or isinstance(tokenizer, GPT2Tokenizer):
        for token in vocab:
            word = tokenizer.decode([vocab[token]])
            if " " in word:
                word = word.strip()
                vocab2idx[word] = vocab[token]
    elif isinstance(tokenizer, XLNetTokenizer)\
            or isinstance(tokenizer, T5Tokenizer):
        for token in vocab:
            if "▁" in token:
                word = tokenizer.decode([vocab[token]]).strip()
                vocab2idx[word] = vocab[token]
    return vocab2idx


def store_vocab2idx(tokenizer, vocab2idx_file):
    vocab2idx = get_whole_vocab2idx(tokenizer)
    store_json_dic(vocab2idx_file, vocab2idx)


def logit_to_list(logits):
    return logits.detach().cpu().numpy().tolist()


def list_filter(lst, index):
    return [lst[i] for i in index]


def list_topk(lst, topk):
    return heapq.nlargest(topk, range(len(lst)), lst.__getitem__)


def read_prompts(relation_id, ori_prompt=None):
    file_path = "fact_data/relations/multi_prompts/{}.jsonl".format(relation_id)
    if ori_prompt:
        relation_prompts = [ori_prompt]
    else:
        relation_prompts = []
    if os.path.isfile(file_path):
        prompts = load_file(file_path)
        for prompt in prompts:
            prompt = prompt["pattern"]
            if prompt not in relation_prompts:
                if "[Y]" not in prompt:
                    print(prompt)
                    continue
                relation_prompts.append(prompt)
    return relation_prompts


def results2distribution(predict_results):
    distribution = {}
    for i in range(len(predict_results)):
        predict_tokens = predict_results[i]['predict_tokens']
        topk_tokens = predict_tokens[: 1]
        for token in topk_tokens:
            if token not in distribution:
                distribution[token] = 0
            distribution[token] += 1
    return distribution


def partition(ls: list, size: int):
    return [ls[i: i+size] for i in range(0, len(ls), size)]


def calculate_p(results):
    n = len(results)
    p = sum([results[i]["ans"] for i in range(n)])
    p = round(p * 100 / n, 2)
    return p


def cal_mean_and_std(lst, central=None):
    mean = round(float(np.mean(lst)), 2)
    if central is None:
        var = round(float(np.var(lst)), 4)
    else:
        var = np.sum([(x-central)**2 for x in lst]) / len(lst)
        var = round(var, 4)
        # std = np.sqrt(var)
        # std = round(std, 4)
    return mean, var


def build_tokenizer(model_name):
    model_type = model_prefix(model_name)
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
    else:
        print(model_name)
        raise RuntimeError('model not exsit')
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    elif model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    elif model_type == "gpt2" or model_type == "gpt":
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    elif model_type == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
    elif model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)
    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    else:
        raise RuntimeError("wrong model")

    return tokenizer


# 按照a的值给两个一起排序
def sync_sort(lst_a, lst_b):
    la, lb = (list(t) for t in zip(*sorted(zip(lst_a, lst_b), reverse=True)))
    return la, lb

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
]


def main():
    data = [[0.1, 0.5], [-0.2], [-0.9]]
    row_labels = ['SM', 'OM']
    col_labels = ['PS', 'RS']
    draw_heat_map(data, row_labels, col_labels)


if __name__ == '__main__':
    main()
