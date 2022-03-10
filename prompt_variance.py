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


def get_model_prompt_mention(model_names):
    model_prompt_mention = {}
    for model_name in model_names:
        data_dir = "fact_data/mention2prediction"
        data_path = "{}/{}".format(data_dir, model_name)
        model_prompt_mention[model_name] = load_json_dic(data_path)
    return model_prompt_mention


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


def cal_precision_stat(lst: list):
    max_p = np.max(lst)
    min_p = np.min(lst)
    std = round(float(np.std(lst)), 2)
    diff = round(max_p - min_p, 2)
    return min_p, max_p, diff, std


def cal_prompt_variance(args):
    fout = open("output/prompt_preference/prompt2precion.txt", "w")
    model_names = [
        "bert-large-cased",
        "roberta-large",
        "gpt2-xl",
        "bart-large",
    ]
    set_seed(args.seed)

    args = get_relation_args(args)

    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, id2samples = lama_data.get_samples()

    model_prompt_mention = get_model_prompt_mention(model_names)

    prompt_choice = "original"
    mention_choice = "original"

    relations = [relation_id for relation_id in id2relation]

    main_tables = {}
    for model_name in model_names:
        table = PrettyTable(field_names=[
            "id", "relation", "worst", "best", "diff", "std"
        ])
        table.title = "prompt for {}".format(model_name)
        main_tables[model_name] = table

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        prompts = read_prompts(relation_id)

        table = PrettyTable(
            field_names=["prompt"] + model_names
        )
        table.title = "prompt precision id: {}, label: {}".format(relation_id, relation_label)

        precisions = {}

        for prompt in prompts:
            new_row = [prompt]
            for model_name in model_names:
                p = cal_mention_precision(
                        model_prompt_mention[model_name],
                        relation_id, prompt,
                        mention_choice=mention_choice
                )
                new_row.append(p)
                if model_name not in precisions:
                    precisions[model_name] = [p]
                else:
                    precisions[model_name].append(p)
            table.add_row(new_row)
        
        for model_name in model_names:
            min_p, max_p, diff, std = cal_precision_stat(precisions[model_name])
            main_tables[model_name].add_row([
                relation_id, relation_label, min_p, max_p, diff, std
            ])
        table = get_table_stat(table)
        print(table)
        print("\n")
        fout.write(table.get_string() + "\n")
    for model_name in model_names:
        table = get_table_stat(main_tables[model_name])
        fout.write(table.get_string() + "\n")
        print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_filter")
    parser.add_argument("--model-name", type=str, default="gpt2-medium")
    parser.add_argument("--model-type", type=str, default="bert")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda-device", type=int, default=0)
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
    parser.add_argument("--sample-times", type=int, default=5)
    parser.add_argument("--sample-num", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-stop-words", action="store_false")

    args = parser.parse_args()
    cal_prompt_variance(args)

if __name__ == '__main__':
    main()