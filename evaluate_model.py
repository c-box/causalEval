import argparse

import torch.cuda
from prettytable import PrettyTable
from utils.utils import set_seed, \
    get_table_stat, get_relation_args, read_prompts
import numpy as np
from models import build_model_wrapper
from utils.read_data import LamaDataset
import random
from tqdm import tqdm
from transformers import BertForMaskedLM
import os


def model_evaluation(args):
    set_seed(0)
    args = get_relation_args(args)
    print(args)
    if args.model_path is not None:
        model_type = args.model_path.split("/")[-1]
        fout = open("{}/{}".format(args.out_dir, model_type), "w")
    else:
        model_type = args.model_name
        fout = open("{}/{}".format(args.out_dir, model_type), "w")

    device_num = torch.cuda.device_count()
    device = random.choice(range(device_num))
    model_wrapper = build_model_wrapper(
        model_name=args.model_name, model_path=args.model_path, args=args, device=args.cuda_device
    )

    lamadata = LamaDataset(relation_file=args.relation_file,
                           sample_dir=args.sample_dir,
                           sample_file_type=args.sample_file_type)

    id2relation, id2samples = lamadata.get_samples()

    table = PrettyTable(
        field_names=["id", "relation", "p"]
    )

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]

        relation_prompts = read_prompts(relation_id)

        samples = id2samples[relation_id]
        results, p = model_wrapper.eval_sample_with_multi_prompts(
            [relation_prompts[0]], samples,
            batch_size=args.batch_size,
            ignore_stop_word=args.ignore_stop_words
        )
        table.add_row([relation_id, relation_label, p])
        print(table)
    table = get_table_stat(table)
    print(table)
    fout.write(table.get_string() + '\n')


def dynamic_evaluation(args):
    set_seed(0)
    args = get_relation_args(args)
    print(args)
    checkpoint_path = "checkpoint/seed_0"
    fout = open("dynamic_results/checkpoint.csv", "w")

    ww = os.walk(checkpoint_path)
    checkpoint_paths = []
    points = [i for i in range(0, 200000, 20000)] + \
        [i for i in range(200000, 2000001, 100000)]
    print(points)
    for point in points:
        cur_path = os.path.join(checkpoint_path, "step_{}".format(point))
        print(cur_path)
        checkpoint_paths.append(cur_path)
    
    model_wrappers = []

    for model_path in checkpoint_paths:
        device_num = torch.cuda.device_count()
        device = random.choice(range(device_num))
        model_wrapper = build_model_wrapper(
            model_name=args.model_name, model_path=model_path, args=args, device=device
        )
        model_wrappers.append(model_wrapper)
    
    lamadata = LamaDataset(relation_file=args.relation_file,
                        sample_dir=args.sample_dir,
                        sample_file_type=args.sample_file_type)

    id2relation, id2samples = lamadata.get_samples()

    pp = ["p_{}".format(i) for i in range(1, len(checkpoint_paths) + 1)]

    table = PrettyTable(
        field_names=["id", "relation"] + pp
    )

    for relation_id in id2relation:
        relation = id2relation[relation_id]
        relation_label = relation["label"]

        relation_prompts = read_prompts(relation_id)
        samples = id2samples[relation_id]

        new_row = [relation_id, relation_label]

        for model_wrapper in model_wrappers:
            results, p = model_wrapper.evaluate_samples(
                    relation, samples,
                    max_len=args.max_len,
                    batch_size=args.batch_size,
                    ignore_stop_word=args.ignore_stop_words
                )
            new_row.append(p)
        table.add_row(new_row)
        print(table)
    table = get_table_stat(table)
    print(table)
    fout.write(table.get_csv_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-type", type=str, default="lama_filter")
    parser.add_argument("--model-name", type=str, default="bert-base-cased")
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
    parser.add_argument("--out-dir", type=str, default="output/sample_disparity")

    parser.add_argument("--task", type=str,
                        default="dynamic_evaluation",
                        choices=[
                            "data_evaluation",
                            "dynamic_evaluation"
                        ])

    parser.add_argument("--ignore-stop-words", action="store_false")

    args = parser.parse_args()
    if args.task == "data_evaluation":
        model_evaluation(args)
    elif args.task == "dynamic_evaluation":
        dynamic_evaluation(args)


if __name__ == '__main__':
    main()
