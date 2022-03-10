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

def cal_prompt_mention_prediction(args):
    if args.model_type == "bert":
        model_names = [
            "bert-large-cased",
            "bert-base-cased"
        ]
        batch_size = 96
    elif args.model_type == "gpt2":
        model_names = [
            "gpt2-xl",
            "gpt2-medium"
        ]
        batch_size = 16
    elif args.model_type == "roberta":
        model_names = [
            "roberta-large",
            "roberta-base"
        ]
        batch_size = 96
    elif args.model_type == "bart":
        model_names = [
            "bart-large",
            "bart-base"
        ]
        batch_size = 96
    model_wrappers = []
    for model_name in model_names:
        model_wrappers.append(
            build_model_wrapper(model_name, device=args.cuda_device, args=args)
        )
    
    set_seed(0)

    args = get_relation_args(args)
    lama_data = LamaDataset(relation_file=args.relation_file,
                            sample_dir=args.sample_dir,
                            sample_file_type=args.sample_file_type)
    id2relation, id2samples = lama_data.get_samples()

    data_dir = "fact_data/mention2prediction"

    for model_name, model_wrapper in zip(model_names, model_wrappers):
        prediction_path = "{}/{}".format(data_dir, model_name)
        mention2prediction = {}
        for relation_id in id2relation:
            mention2prediction[relation_id] = {}
            relation_prompts = read_prompts(relation_id)
            samples = id2samples[relation_id]
            relation = id2relation[relation_id]
            relation_label = relation["label"]
            expand_samples = []
            for sample in samples:
                sub, obj, sub_id = get_pair(sample, return_id=True)
                sub_mentions = sample["sub_mentions"]
                for mention in sub_mentions:
                    new_sample = copy.deepcopy(sample)
                    sample["sub_label"] = mention
                    expand_samples.append(new_sample)
            for prompt in tqdm(relation_prompts):
                results, p, tokens, pre_res = model_wrapper.eval_sample_with_multi_prompts(
                    [prompt], expand_samples,
                    batch_size=batch_size,
                    ignore_stop_word=args.ignore_stop_words, 
                    return_tokens=True
                )
                mention2prediction[relation_id][prompt] = {}
                for token, res, sample in zip(tokens, pre_res, expand_samples):
                    sub, obj, sub_id = get_pair(sample, return_id=True)
                    if sub_id not in mention2prediction[relation_id][prompt]:
                        mention2prediction[relation_id][prompt][sub_id] = {}
                    mention2prediction[relation_id][prompt][sub_id][sub] = {
                        "prediction": token, "res": res
                    }
            prompt = "multi"
            this_batch = int(batch_size / len(relation_prompts))
            if this_batch < 1:
                this_batch = 1
            results, p, tokens, pre_res = model_wrapper.eval_sample_with_multi_prompts(
                    relation_prompts, expand_samples,
                    batch_size=this_batch,
                    ignore_stop_word=args.ignore_stop_words, 
                    return_tokens=True
                )
            mention2prediction[relation_id][prompt] = {}
            for token, res, sample in zip(tokens, pre_res, expand_samples):
                sub, obj, sub_id = get_pair(sample, return_id=True)
                if sub_id not in mention2prediction[relation_id][prompt]:
                    mention2prediction[relation_id][prompt][sub_id] = {}
                mention2prediction[relation_id][prompt][sub_id][sub] = {
                    "prediction": token, "res": res
                }
        store_json_dic(prediction_path, mention2prediction)
                    



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
    parser.add_argument("--sample-times", type=int, default=5)
    parser.add_argument("--sample-num", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--task", type=str,
                        default="cal_prompt_mention_prediction",
                        choices=[
                            "cal_prompt_mention_prediction"
                        ])

    parser.add_argument("--ignore-stop-words", action="store_false")

    args = parser.parse_args()

    if args.task == "cal_prompt_mention_prediction":
        cal_prompt_mention_prediction(args)


if __name__ == '__main__':
    main()
