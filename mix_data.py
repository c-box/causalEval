from json import load
from utils.utils import load_file, set_seed, store_file
import os
from tqdm import tqdm
import argparse
import random
from utils.read_data import LamaDataset
import lzma


def load_lama_ctx(
        relation_file="fact_data/relations/relations_without_nm.jsonl",
        sample_dir="fact_data/filter_TREx"):
    lamadata = LamaDataset(relation_file=relation_file,
                           sample_dir=sample_dir,
                           sample_file_type="jsonl")

    id2relation, id2samples = lamadata.get_samples()
    sents = []
    for relation_id in id2relation:
        relation = id2relation[relation_id]
        data = id2samples[relation_id]
        for sample in data:
            ctxs = sample["evidences"]
            ss = []
            for ctx in ctxs:
                sub = ctx["sub_surface"]
                obj = ctx["obj_surface"]
                sent = ctx["masked_sentence"]
                sent = sent.replace("[MASK]", obj)
                if sent in ss:
                    continue
                ss.append(sent)
                sents.append(sent)
    return sents


def store_lama_ctx(store_file="data/lama_sents.txt"):
    with open(store_file, "w") as f:
        sents = load_lama_ctx()
        f.write("\n".join(sents))
        print("num of data {}".format(len(sents)))


def read_lama_sents(data_file="data/lama_sents.txt"):
    with open(data_file, "r") as f:
        sents = f.read().split("\n")
        print("lama_num: {}".format(len(sents)))
        return sents


def read_webtext_sents(data_file="data/webtext_sents.txt"):
    with open(data_file, "r") as f:
        sents = f.read().split("\n")
        print("webtext_num: {}".format(len(sents)))
        return sents


def read_bookcorpus_sents(data_file="data/BookCorpus/books_large_p1.txt"):
    with open(data_file, "r") as f:
        sents = f.read().split("\n")
        print("book_num: {}".format(len(sents)))
        return sents


def store_webtext(data_dir="data/openwebtext"):
    sents = []
    for idx in tqdm(range(1, 1001)):
        data_file = "{}/urlsf_subset00-{}_data.xz".format(data_dir, idx)
        with lzma.open(data_file, "r") as f:
            for s in f:
                if s == "\n":
                    continue
                sent = s.decode(encoding="utf-8", errors="ignore").strip("\n").strip(" ")
                if len(sent.split()) < 8:
                    continue
                sents.append(sent)
    with open("data/webtext_sents.txt", "w") as f:
        f.write("\n".join(sents))
        print("num of data: {}".format(len(sents)))


def mix_data(data_path="pretrain_data", mix_method="replace", mix_alpha=0.0):
    set_seed(0)
    data_file = "{}/{}_{}.txt".format(data_path, mix_method, mix_alpha)
    with open(data_file, "w") as f:
        lama_sents = read_lama_sents()
        webtext_sents = read_webtext_sents()
        lama_num = len(lama_sents)
        if mix_method == "replace":
            if mix_alpha == "test":
                mix_num = 1000
                original_num = 0
            else:
                original_num = int(lama_num * mix_alpha)
                mix_num = lama_num - original_num
            sampled_lama = random.sample(lama_sents, original_num)
            sampled_book = random.sample(webtext_sents, mix_num)
            mix_sents = sampled_lama + sampled_book
            print("mix_sents_num:{}".format(len(mix_sents)))
            random.shuffle(mix_sents)
            f.write("\n".join(mix_sents))


def copy_data(data_path="pretrain_data", mix_method="replace", mix_alpha=0.0, dupe=3):
    set_seed(0)
    data_file = "{}/{}_{}.txt".format(data_path, mix_method, mix_alpha)
    with open(data_file, "r") as f:
        data = f.read().split("\n")
    dupe_data = data * dupe
    dupe_data_file = "{}/{}*{}_{}.txt".format(data_path, dupe, mix_method, mix_alpha)
    with open(dupe_data_file, "w") as fout:
        fout.write("\n".join(dupe_data))
    print(len(dupe_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-method", type=str, default="replace", choices=["replace", "add"])
    parser.add_argument("--mix-alpha", type=float, default="0.4")
    args = parser.parse_args()
    store_lama_ctx()
    store_webtext()
    mix_data(mix_method=args.mix_method, mix_alpha=args.mix_alpha)
    # copy_data(mix_method=args.mix_method, mix_alpha=args.mix_alpha, dupe=3)


if __name__ == '__main__':
    main()
