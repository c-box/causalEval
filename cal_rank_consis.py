import os

import torch
from prettytable import PrettyTable
from utils.utils import set_seed, \
    get_table_stat, model_prefix, get_relation_args, \
    load_file, partition, calculate_p, cal_mean_and_std, get_pair, \
    read_prompts, store_json_dic, load_json_dic
import numpy as np
from models import build_model_wrapper, model_wrapper
from utils.read_data import LamaDataset
import random
from tqdm import tqdm
from transformers import GPT2Tokenizer
import argparse
import copy

def get_rank_consis(model_sorts, model_names):
    sample_nums = len(model_sorts)
    model_num = len(model_names)
    model_rank_consis = []
    model_rank_std = []
    for model_idx in range(model_num):
        consis = [0 for i in range(model_num)]
        ranks = []
        for i in range(sample_nums):
            rank = np.where(model_sorts[i]==model_idx)[0][0]
            consis[rank] += 1
            ranks.append(rank)
        max_consis = max(consis)
        model_rank_consis.append(round(max_consis*100/sample_nums, 2))
        rank_std = np.std(ranks)
        rank_std = round(rank_std, 2)
        model_rank_std.append(rank_std)
    return model_rank_consis, model_rank_std


def get_all_rank_consis(model_sorts):
    sample_num = len(model_sorts)
    consis = [0 for i in range(sample_num)]
    for i in range(sample_num):
        for j in range(sample_num):
            if (model_sorts[i] == model_sorts[j]).all():
                consis[i] += 1
    max_consis = max(consis)
    return round(max_consis*100/sample_num, 2)


def get_model_prompt_mention(model_names):
    model_prompt_mention = {}
    for model_name in model_names:
        data_dir = "fact_data/mention2prediction"
        data_path = "{}/{}".format(data_dir, model_name)
        model_prompt_mention[model_name] = load_json_dic(data_path)
    return model_prompt_mention


def final_rank_consis(args):
    sample_times = args.sample_times
    sample_num = args.sample_num
    fout = open("output/causal_intervention/final_rank_consis", "w")
    model_names = [
        "bert-base-cased",
        "bert-large-cased",
        "roberta-base",
        "roberta-large",
        "gpt2-medium",
        "gpt2-xl",
        "bart-base",
        "bart-large",
    ]
    set_seed(args.seed)

    args = get_relation_args(args)

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, id2samples = lama_data.get_samples()

    model_prompt_mention = get_model_prompt_mention(model_names)
    print("results loaded")
    choices = ["original", "random", "multi"]
    table = PrettyTable(field_names=["choice"] + model_names + ["all"])
    relation_ids = [relation_id for relation_id in id2relation]
    model_choice2sorts = {"original": [], "random": [], "multi": []}
    for sample_time in tqdm(range(sample_times)):
        relations = random.sample(relation_ids, sample_num)
        for choice in choices:
            ps = []
            for model_name in model_names:
                p = cal_prompt_mention_precision(
                    relations, model_prompt_mention[model_name],
                    prompt_choice=choice, mention_choice=choice
                )
                ps.append(p)
            p_sort = np.argsort(ps)
            model_choice2sorts[choice].append(p_sort)

    for choice in choices:
        model_sorts = model_choice2sorts[choice]
        model_rank_consis, model_rank_std = get_rank_consis(model_sorts, model_names)
        all_model_rank = get_all_rank_consis(model_sorts)
        table.add_row([choice] + model_rank_consis + [all_model_rank])
    print(table)
    fout.write(table.get_string() + "\n")

def cal_mention_precision(prompt2precision, relation_id, single_prompt,mention_choice ):
    sub_ids = [key for key in prompt2precision[relation_id][single_prompt]]
    total = len(sub_ids)
    p = 0
    for sub_id in sub_ids:
        mentions = [mention for mention in prompt2precision[relation_id][single_prompt][sub_id]]
        if mention_choice == "multi":
            pp = 0
            for mention in mentions:
                result = prompt2precision[relation_id][single_prompt][sub_id][mention]
                if result["res"]:
                    pp += 1
            pp = pp / len(mentions)
            p += pp
        else:
            if mention_choice == "random":
                mention = random.choice(mentions)
            elif mention_choice == "original":
                mention = mentions[0]
            result = prompt2precision[relation_id][single_prompt][sub_id][mention]
            if result["res"]:
                p += 1
    p = round(p * 100 / total, 2)
    return p


def cal_prompt_mention_precision(
    relations, prompt2precision, prompt_choice="random", mention_choice="random"
    ):
    ps = []
    for relation_id in relations:
        relation_prompts = read_prompts(relation_id)
        if prompt_choice == "random":
            single_prompt = random.choice(relation_prompts)
            p = cal_mention_precision(prompt2precision, relation_id, single_prompt,mention_choice)
        elif prompt_choice == "multi":
            prompt_ps = []
            for prompt in relation_prompts:
                prompt_p = cal_mention_precision(prompt2precision, relation_id, prompt, mention_choice)
                prompt_ps.append(prompt_p)
            p = np.mean(prompt_ps)
        elif prompt_choice == "original":
            single_prompt = relation_prompts[0]
            p = cal_mention_precision(prompt2precision, relation_id, single_prompt,mention_choice)
        elif prompt_choice == "ensemble":
            single_prompt = "multi"
            p = cal_mention_precision(prompt2precision, relation_id, single_prompt,mention_choice)
        else:
            raise RuntimeError("no such xxx")
        ps.append(p)
    # print(ps)
    mean_p = np.mean(ps)
    mean_p = round(float(mean_p), 2)
    return mean_p
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_filter")
    parser.add_argument("--model-name", type=str, default="gpt2-medium")
    parser.add_argument("--model-type", type=str, default="bert")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--gpt-method", type=str, default="next_token")
    parser.add_argument("--generate-len", type=int, default=1)

    parser.add_argument("--sample-method", type=str, default="replace",
                        choices=["replace", "no_replace"])

    parser.add_argument("--dupe", type=int, default=5)
    parser.add_argument("--lr", type=str, default="5e-5")
    parser.add_argument("--model-path", type=str, default=None)

    parser.add_argument("--multi-prompt", type=bool, default=True)
    parser.add_argument("--sample-times", type=int, default=1000)
    parser.add_argument("--sample-num", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--task", type=str,
                        default="final_rank_consis",
                        choices=[
                            "cal_rank_consis_with_mention",
                            "cal_mention_consis",
                            "cal_mention_performance_variance",
                            "final_rank_consis"
                        ])

    parser.add_argument("--ignore-stop-words", action="store_false")

    args = parser.parse_args()

    if args.task == "final_rank_consis":
        final_rank_consis(args)

if __name__ == '__main__':
    main()

