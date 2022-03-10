import random

from prettytable import PrettyTable
from utils.utils import get_relation_args, get_table_stat, get_pair, set_seed, model_prefix
from utils.constant import stop_words
from models import build_model_wrapper
from utils.read_data import LamaDataset
from utils.utils import set_plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse

def eval_mention_consistency(args):
    fout = open("output/instance_verb/mention_consistent_{}".format(args.model_name), "w", encoding="utf-8")
    set_seed(0)
    args = get_relation_args(args)
    model_wrapper = build_model_wrapper(model_name=args.model_name, device=args.cuda_device, args=args)

    lamadata = LamaDataset(relation_file=args.relation_file,
                           sample_dir=args.sample_dir,
                           sample_file_type=args.sample_file_type)

    id2relation, id2samples = lamadata.get_samples()

    table = PrettyTable(field_names=[
        "id", "relation", "mention_consis"
    ])

    table.title = "{}".format(args.model_name)

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]
        samples = id2samples[relation_id]

        mention2sample = {}
        consis = 0
        total = 0

        for sample in samples:
            sub_mentions = sample["sub_mentions"]
            num_mention = len(sub_mentions)
            if num_mention > 5:
                num_mention = 5
            if num_mention not in mention2sample:
                mention2sample[num_mention] = [sample]
            else:
                mention2sample[num_mention].append(sample)

        for num_mention in mention2sample:
            if num_mention <= 1:
                continue
            samples = mention2sample[num_mention]
            res = [[] for i in range(len(samples))]
            total += len(samples)
            flag = [1 for i in range(len(samples))]
            for idx in range(0, num_mention):
                for i in range(len(samples)):
                    samples[i]["sub_label"] = samples[i]["sub_mentions"][idx]
                results, p = model_wrapper.evaluate_samples(
                    relation, samples,
                    batch_size=args.batch_size, max_len=args.max_len,
                    ignore_stop_word=args.ignore_stop_words
                )
                for i in range(len(samples)):
                    token = results[i]["predict_tokens"][0]
                    if args.ignore_stop_words:
                        k = 1
                        while token in stop_words \
                                and k < len(results[i]["predict_tokens"]):
                            token = results[i]["predict_tokens"][k]
                            k += 1
                    res[i].append(token)
            for i in range(len(samples)):
                for j in range(num_mention - 1):
                    if res[i][j] != res[i][j+1]:
                        flag[i] = 0
                        break
            for ff in flag:
                consis += ff

        table.add_row([relation_id, relation_label, round(consis * 100 / total, 2)])
        print(table)
    table = get_table_stat(table)
    print(table)
    fout.write(table.get_string() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_filter")
    parser.add_argument("--model-name", type=str, default="gpt2-xl",
                        choices=["bert-large-cased", "roberta-large", "gpt2-large",
                                 "gpt2-xl", "bart-large"])

    parser.add_argument("--gpt-method", type=str, default="next_token")
    parser.add_argument("--generate-len", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--ignore-stop-words", action="store_false")

    parser.add_argument("--task", type=str,
                        default="eval_mention_consistency")

    args = parser.parse_args()

    eval_mention_consistency(args)


if __name__ == '__main__':
    main()
