from utils.constant import CUDA_DEVICE, BATCH_SIZE
from utils.utils import get_pair, store_vocab2idx, load_json_dic, stop_words
from transformers import PreTrainedTokenizer, PreTrainedModel
from models.model_wrapper import ModelWrapper
from models.mlm_wrapper import MLMWrapper
import torch
from tqdm import tqdm
import os
import numpy as np


class GPTWrapper(MLMWrapper):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 gpt_method="next_token",
                 generate_len=1,
                 vocab2idx_file="data/gpt2_data/vocab2idx.json",
                 device: int = None):
        super().__init__(tokenizer, model, device)
        self.gpt_method = gpt_method
        self.generate_len = generate_len
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if not os.path.isfile(vocab2idx_file):
        #     store_vocab2idx(tokenizer, vocab2idx_file)
        # self.vocab2idx = load_json_dic(vocab2idx_file)

    def token_to_idx(self, token):
        return self.vocab2idx[token]

    def get_predict_score(self,
                          input_text: list,
                          max_len=256
                          ):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.batch_encode_plus(
            input_text, padding='longest',
            truncation=True, max_length=max_len,
            return_tensors="pt", add_special_tokens=True
        )
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs, return_dict=True)
        prediction_score = outputs.logits
        return prediction_score

    def predict(self,
                input_texts: list,
                mask_pos=-1,
                batch_size=BATCH_SIZE,
                obj_tokens=None,
                topk=10,
                rank_k=10000,
                max_len=256
                ):
        """
        :param input_texts:
        :param mask_pos:
        :param batch_size:
        :param obj_tokens:
        :param topk:
        :param rank_k:
        :param max_len:
        :return: 下一个单词的预测结果（只有一个单词）
        """
        batch_text = self.partition(input_texts, batch_size)
        predict_results = []

        if obj_tokens is None:
            for input_text in batch_text:
                predict_logits = self.get_predict_score(input_text, max_len=max_len)

                for i in range(len(input_text)):
                    text = input_text[i]
                    prompt_length = len(self.tokenizer.encode(text, add_special_tokens=False))
                    logits = predict_logits[i][prompt_length-1]
                    predicted_tokens, predicted_prob = self.logits_to_results(
                        logits, topk=topk
                    )
                    predict_results.append({'predict_tokens': predicted_tokens,
                                            'predict_prob': predicted_prob})
        else:
            assert len(obj_tokens) == len(input_texts)
            batch_obj = self.partition(obj_tokens, batch_size)

            for input_text, objs in zip(batch_text, batch_obj):
                predict_logits = self.get_predict_score(input_text, max_len=max_len)
                for i in range(len(input_text)):
                    text = input_text[i]
                    prompt_length = len(self.tokenizer.encode(text, add_special_tokens=False))
                    logits = predict_logits[i][prompt_length - 1]
                    obj = objs[i]
                    predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr = \
                        self.logits_to_results_with_obj(
                            logits, topk, obj, rank_k=rank_k
                        )
                    if obj == predicted_tokens[0]:
                        predict_ans = True
                    else:
                        predict_ans = False
                    predict_results.append({'predict_tokens': predicted_tokens,
                                            'predict_prob': predicted_prob,
                                            'obj_prob': obj_prob,
                                            'obj_rank': obj_rank,
                                            'predict_ans': predict_ans,
                                            'mrr': mrr})
        return predict_results

    def gpt_generate(self, input_text, max_len=256):
        inputs = self.tokenizer.batch_encode_plus(
            input_text, padding='longest',
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        # 参数可修改
        outputs = self.model.generate(
            **inputs, do_sample=False, max_length=max_len, pad_token_id=self.tokenizer.eos_token_id
        )
        return outputs

    # 只能返回一个结果，没有排名信息等
    def predict_sequence(self, input_texts, batch_size=BATCH_SIZE):
        batch_text = self.partition(input_texts, batch_size)
        self.tokenizer.padding_side = "left"

        predict_results = []
        input_text_len = [len(self.tokenizer.encode(t)) for t in input_texts]
        max_len = max(input_text_len) + 8

        for input_text in tqdm(batch_text):
            outputs = self.gpt_generate(input_text, max_len=max_len)
            outputs = self.tokenizer.batch_decode(outputs,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            for i in range(len(outputs)):
                prompt_length = len(input_text[i].split())
                predict_tokens = outputs[i].split()
                predict_token = predict_tokens[prompt_length]
                predict_seq = predict_tokens[prompt_length:]
                predict_results.append({"predict_tokens": [predict_token], "predict_seq": predict_seq})
        return predict_results

    def prompt_to_sent(self, prompt: str, sub, obj=None, predict_sub=False):
        assert "[X]" in prompt
        assert "[Y]" in prompt
        sent = prompt.replace("[X]", sub)
        sent = sent.replace("[Y]", "").strip(".").strip(" ")
        return sent

    def evaluate_samples(self, relation, samples,
                         pass_obj=False,
                         ignore_stop_word=False,
                         **kwargs):
        relation_prompt = relation["template"]
        input_texts = []
        gold_obj = []
        p_1 = 0

        for sample in samples:
            sub, obj = get_pair(sample)
            sent = self.prompt_to_sent(relation_prompt, sub, obj)
            prompt_length = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if prompt_length >= 256:
                continue
            gold_obj.append(obj)
            input_texts.append(sent)

        if self.gpt_method == "next_token":
            predict_results = self.predict(
                input_texts,
                obj_tokens=gold_obj if pass_obj else None,
                **kwargs
            )
        elif self.gpt_method == "generate_sequence":
            predict_results = self.predict_sequence(
                input_texts,
                batch_size=kwargs["batch_size"] if "batch_size" in kwargs else BATCH_SIZE
            )
        else:
            raise RuntimeError("wrong method")

        for i in range(len(predict_results)):
            if self.gpt_method == "generate_sequence":
                predict_tokens = " ".join(predict_results[i]["predict_seq"][: self.generate_len])
            else:
                predict_tokens = predict_results[i]["predict_tokens"][0]
                if ignore_stop_word:
                    k = 1
                    while predict_tokens in stop_words \
                            and k < len(predict_results[i]["predict_tokens"]):
                        predict_tokens = predict_results[i]["predict_tokens"][k]
                        k += 1
            obj = gold_obj[i]
            if obj in predict_tokens:
                p_1 += 1
        if len(gold_obj) == 0:
            p_1 = 0
        else:
            p_1 = round(p_1 * 100 / len(gold_obj), 2)

        return predict_results, p_1

    def get_predict_score_for_gpt(self,
                                  input_text: list,
                                  max_len=256
                                  ):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.batch_encode_plus(
            input_text, padding='longest',
            truncation=True, max_length=max_len,
            return_tensors="pt", add_special_tokens=True
        )
        mask_index = []
        for i in range(len(input_text)):
            mask_index.append(
                [len(self.tokenizer.encode(input_text[i], add_special_tokens=False))-1]
            )
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs, return_dict=True)
        prediction_score = outputs.logits
        return prediction_score, mask_index

    def get_mean_logits(self, mask_pos, mask_index, logits):
        obj_pos = torch.tensor([mask_index[i][mask_pos[i]] for i in range(len(mask_index))]).cuda(self.device)
        obj_pos = torch.reshape(obj_pos, (logits.shape[0], 1, 1))
        index = obj_pos.expand(logits.shape[0], 1, logits.shape[-1])
        predict_logits = torch.gather(logits, 1, index).reshape((logits.shape[0], logits.shape[-1]))
        # prompt_logits = torch.index_select(logits, 1, obj_pos)
        mean_logits = torch.mean(predict_logits, dim=0)
        return mean_logits

    def eval_sample_with_multi_prompts(
            self, relation_prompts, samples, batch_size, topk=10, max_len=256, ignore_stop_word=True,return_tokens=False
    ):
        prompt_num = len(relation_prompts)
        input_texts = []
        objs = []
        for sample in samples:
            sub, obj = get_pair(sample)
            objs.append(obj)
            for prompt in relation_prompts:
                sent = self.prompt_to_sent(prompt, sub, obj)
                input_texts.append(sent)

        partition_size = batch_size * prompt_num
        batch_text = self.partition(input_texts, partition_size)
        predict_results = []

        for idx in range(len(batch_text)):
            single_batch_text = batch_text[idx]
            this_batch = int(len(single_batch_text) / prompt_num)
            mask_pos = [-1 for i in range(prompt_num)]
            predict_logits, mask_index = self.get_predict_score_for_gpt(
                input_text=single_batch_text, max_len=max_len
            )
            mask_indexs = self.partition(mask_index, prompt_num)
            # print(predict_logits.shape)
            predict_logits = torch.reshape(
                predict_logits,
                (this_batch, prompt_num,
                 predict_logits.shape[-2], predict_logits.shape[-1]
                 ))
            # print(predict_logits.shape)
            # print("")

            for instance_idx in range(this_batch):
                mean_logits = self.get_mean_logits(
                    mask_pos, mask_indexs[instance_idx],
                    predict_logits[instance_idx]
                )
                predicted_tokens, predicted_prob = self.logits_to_results_without_softmax(
                    mean_logits, topk=topk
                )
                predict_results.append({'predict_tokens': predicted_tokens,
                                        'predict_prob': predicted_prob})

        p_1 = 0
        predict_tokens = []
        predict_res = []
        for i in range(len(predict_results)):
            predict_token = predict_results[i]["predict_tokens"][0]
            if ignore_stop_word:
                k = 1
                while predict_token in stop_words \
                        and k < len(predict_results[i]["predict_tokens"]):
                    predict_token = predict_results[i]["predict_tokens"][k]
                    k += 1
            predict_tokens.append(predict_token)
            if predict_token == objs[i]:
                p_1 += 1
                predict_res.append(True)
            else:
                predict_res.append(False)

        if len(objs) == 0:
            p_1 = 0
        else:
            p_1 = round(p_1 * 100 / len(objs), 2)

        if return_tokens:
            return predict_results, p_1, predict_tokens, predict_res
        else:
            return predict_results, p_1
    
    def eval_sample_with_multi_prompts_and_mention(
            self, relation_prompts, samples, batch_size, topk=10,
            max_len=256, ignore_stop_word=True, mentions=3
    ):
        prompt_num = len(relation_prompts)
        input_texts = []
        objs = []
        for sample in samples:
            sub, obj = get_pair(sample)
            objs.append(obj)
            sub_mentions = sample["sub_mentions"]
            num_mention = len(sub_mentions)
            if mentions == -1:
                used_mentions = np.random.choice(
                    sub_mentions, size=1, replace=True
                )
            else:
                if num_mention < mentions:
                    used_mentions = [sub] + list(np.random.choice(
                        sub_mentions, size=mentions - 1, replace=True
                    ))
                else:
                    used_mentions = [sub] + list(np.random.choice(
                        sub_mentions, size=mentions - 1, replace=False
                    ))
            for prompt in relation_prompts:
                for mention in used_mentions:
                    sent = self.prompt_to_sent(prompt, mention, obj)
                    input_texts.append(sent)

        if mentions != -1:
            partition_size = batch_size * prompt_num * mentions
            prompt_num = prompt_num * mentions
        else:
            partition_size = batch_size * prompt_num

        batch_text = self.partition(input_texts, partition_size)
        predict_results = []

        for idx in range(len(batch_text)):
            single_batch_text = batch_text[idx]
            this_batch = int(len(single_batch_text) / prompt_num)
            mask_pos = [-1 for i in range(prompt_num)]
            predict_logits, mask_index = self.get_predict_score_for_gpt(
                input_text=single_batch_text, max_len=max_len
            )
            mask_indexs = self.partition(mask_index, prompt_num)
            # print(predict_logits.shape)
            predict_logits = torch.reshape(
                predict_logits,
                (this_batch, prompt_num,
                 predict_logits.shape[-2], predict_logits.shape[-1]
                 ))
            # print(predict_logits.shape)
            # print("")

            for instance_idx in range(this_batch):
                mean_logits = self.get_mean_logits(
                    mask_pos, mask_indexs[instance_idx],
                    predict_logits[instance_idx]
                )
                predicted_tokens, predicted_prob = self.logits_to_results_without_softmax(
                    mean_logits, topk=topk
                )
                predict_results.append({'predict_tokens': predicted_tokens,
                                        'predict_prob': predicted_prob})

        p_1 = 0
        for i in range(len(predict_results)):
            predict_token = predict_results[i]["predict_tokens"][0]
            if ignore_stop_word:
                k = 1
                while predict_token in stop_words \
                        and k < len(predict_results[i]["predict_tokens"]):
                    predict_token = predict_results[i]["predict_tokens"][k]
                    k += 1
            if predict_token == objs[i]:
                p_1 += 1

        if len(objs) == 0:
            p_1 = 0
        else:
            p_1 = round(p_1 * 100 / len(objs), 2)

        return predict_results, p_1
