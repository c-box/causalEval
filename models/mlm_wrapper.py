from utils.constant import CUDA_DEVICE, BATCH_SIZE
from utils.utils import get_pair, sync_sort, stop_words
from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from models.model_wrapper import ModelWrapper
import torch
from tqdm import tqdm
import numpy as np


class MLMWrapper(ModelWrapper):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: PreTrainedModel,
                 device: int = None):
        super().__init__(tokenizer, model, device)
        self.vocab = tokenizer.get_vocab()

    def get_model_output(self, input_text: list, max_len=256):
        inputs, mask_index = self.get_model_inputs(input_text, max_len)
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        return outputs, mask_index


    def get_model_inputs(self, input_text: list, max_len=256):
        inputs = self.tokenizer.batch_encode_plus(
            input_text, padding="longest", truncation=True, max_length=max_len
        )

        mask_token_id = self.tokenizer.mask_token_id
        mask_index = []
        input_ids = inputs["input_ids"]
        # support multi mask
        # 这里的index可以有多个
        for ids in input_ids:
            index = self.get_index(ids, mask_token_id)
            mask_index.append(index)

        for key in inputs:
            inputs[key] = torch.tensor(inputs[key]).cuda(self.device)
        return inputs, mask_index


    # get the predict logits and position of mask token
    def get_predict_score(self,
                          input_text: list,
                          max_len=256
                          ):
        outputs, mask_index = self.get_model_output(input_text, max_len)
        predict_logits = outputs.logits

        return predict_logits, mask_index

    # 返回mask位置的表示，batch_size * hidden_size
    def get_mask_hidden(self, input_text, max_len=256):
        outputs, mask_index = self.get_model_output(input_text, max_len)
        # 最后一层
        hidden_states = outputs.hidden_states
        # 取一个
        hidden_states = hidden_states[-1].detach()
        obj_pos = torch.tensor([mask_index[i][0] for i in range(len(mask_index))]).cuda(self.device)
        obj_pos = torch.reshape(obj_pos, (hidden_states.shape[0], 1, 1))
        index = obj_pos.expand(hidden_states.shape[0], 1, hidden_states.shape[-1])
        mask_hidden_states = torch.gather(
            hidden_states, 1, index
        ).reshape((hidden_states.shape[0], hidden_states.shape[-1]))
        del hidden_states
        return mask_hidden_states

    def prob_to_tokens(self, predicted_prob, predicted_index):
        predicted_prob = predicted_prob.detach().cpu().numpy()
        predicted_index = predicted_index.cpu().numpy().tolist()
        predicted_tokens = []
        for index in predicted_index:
            predicted_tokens.append(self.tokenizer.decode([index]).strip())
        return predicted_tokens, predicted_prob

    def logits_to_results(self, logits, topk):
        logits = torch.softmax(logits, dim=-1)
        predicted_prob, predicted_index = torch.topk(logits, topk)
        return self.prob_to_tokens(predicted_prob, predicted_index)

    def softmax_to_results(self, logits, prompt_logits, topk):
        logits = torch.softmax(logits, dim=-1)
        logits = logits - prompt_logits
        predicted_prob, predicted_index = torch.topk(logits, topk)
        return self.prob_to_tokens(predicted_prob, predicted_index)

    def logits_to_results_without_softmax(self, logits, topk):
        predicted_prob, predicted_index = torch.topk(logits, topk)
        return self.prob_to_tokens(predicted_prob, predicted_index)

    # token to idx, for roberta and gpt, need to overload
    def token_to_idx(self, token):
        index = self.tokenizer.convert_tokens_to_ids(token)
        return index

    # return the rank and mrr
    def logits_to_results_with_obj(self, logits, topk, obj, rank_k=10000):
        predicted_tokens, predcited_probs = self.logits_to_results(logits, topk)
        logits = torch.softmax(logits, dim=-1)
        obj_index = self.token_to_idx(obj)
        obj_prob = logits[obj_index].item()

        rank_prob, rank_index = torch.topk(logits, rank_k)
        rank_index = rank_index.cpu().numpy().tolist()

        if obj_index not in rank_index:
            obj_rank = rank_k
            mrr = 0
        else:
            obj_rank = rank_index.index(obj_index) + 1
            mrr = 1 / obj_rank

        return predicted_tokens, predcited_probs, obj_prob, obj_rank, mrr

    # return the predict results given the input sentences as input
    def predict(self,
                input_texts: list,
                mask_pos=0,
                batch_size=BATCH_SIZE,
                obj_tokens=None,
                topk=10,
                rank_k=10000,
                max_len=256
                ):
        """
        :param input_texts:
        :param mask_pos: 0 for the fist mask, -1 for the last mask, list for particular assign
        :param batch_size:
        :param obj_tokens: if provide, will return the rank and prob info
        :param topk:
        :param rank_k:
        :param max_len:
        :return: predict_results
        """
        assert isinstance(mask_pos, int) or isinstance(mask_pos, list)
        if isinstance(mask_pos, int):
            mask_pos_lst = [mask_pos] * len(input_texts)
        else:
            mask_pos_lst = mask_pos
        assert len(mask_pos_lst) == len(input_texts)

        batch_text = self.partition(input_texts, batch_size)
        batch_mask_pos = self.partition(mask_pos_lst, batch_size)

        predict_results = []

        if obj_tokens is None:
            for idx in range(len(batch_text)):
                single_batch_text = batch_text[idx]
                single_batch_mask_pos = batch_mask_pos[idx]

                predict_logits, mask_index = self.get_predict_score(
                    single_batch_text, max_len=max_len
                )

                for i in range(len(single_batch_text)):
                    assert isinstance(single_batch_mask_pos[i], int)
                    mask_pos_id = single_batch_mask_pos[i]
                    try:
                        logits = predict_logits[i][mask_index[i][mask_pos_id]]
                    except:
                        print(mask_pos_id)
                        print(mask_index[i])

                    predicted_tokens, predicted_prob = self.logits_to_results(
                        logits, topk=topk
                    )
                    predict_results.append({'predict_tokens': predicted_tokens,
                                            'predict_prob': predicted_prob})
        else:
            assert len(obj_tokens) == len(input_texts)
            batch_obj = self.partition(obj_tokens, batch_size)

            for idx in tqdm(range(len(batch_text))):
                single_batch_text = batch_text[idx]
                single_batch_mask_pos = batch_mask_pos[idx]
                single_batch_obj = batch_obj[idx]

                predict_logits, mask_index = self.get_predict_score(
                    single_batch_text, max_len=max_len
                )

                for i in range(len(single_batch_text)):
                    assert isinstance(single_batch_mask_pos[i], int)
                    mask_pos_id = single_batch_mask_pos[i]
                    logits = predict_logits[i][mask_index[i][mask_pos_id]]
                    obj = single_batch_obj[i]

                    predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr = \
                        self.logits_to_results_with_obj(
                            logits, topk, obj, rank_k=rank_k
                        )

                    if single_batch_obj[i] == predicted_tokens[0]:
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

    # num * hidden_size
    def get_mask_presentation(self, prompt, samples, batch_size=BATCH_SIZE):
        input_texts = []
        for sample in samples:
            sub, obj = get_pair(sample)
            sent = self.prompt_to_sent(prompt, sub, obj)
            input_texts.append(sent)
        batch_text = self.partition(input_texts, batch_size)
        mask_hiddens = []
        for idx in range(len(batch_text)):
            single_batch_text = batch_text[idx]
            batch_mask_hidden_states = self.get_mask_hidden(single_batch_text)
            mask_hiddens.append(batch_mask_hidden_states)
        mask_presentation = torch.cat(tuple(mask_hiddens), 0)
        return mask_presentation

    def prompt_to_sent(self, prompt: str, sub, obj=None, predict_sub=False):
        # print(prompt)
        assert "[X]" in prompt
        assert "[Y]" in prompt
        if predict_sub is False:
            sent = prompt.replace("[X]", sub)
            mask_token = self.tokenizer.mask_token
            sent = sent.replace("[Y]", mask_token)
        else:
            assert obj is not None
            sent = prompt.replace("[Y]", obj)
            mask_token = self.tokenizer.mask_token
            sent = sent.replace("[X]", mask_token)
        return sent

    # one mask only
    def evaluate_samples(self, relation, samples, pass_obj=False,
                         predict_sub=False, ignore_stop_word=False,
                         **kwargs):
        relation_prompt = relation["template"]
        input_texts = []
        gold_ans = []
        p_1 = 0

        for sample in samples:
            if "sub_label" not in sample:
                print(sample)
                continue
            sub, obj = get_pair(sample)
            if predict_sub:
                gold_ans.append(sub)
            else:
                gold_ans.append(obj)
            sent = self.prompt_to_sent(relation_prompt, sub, obj, predict_sub=predict_sub)
            input_texts.append(sent)
        with torch.no_grad():
            if pass_obj:
                predict_results = self.predict(
                    input_texts, obj_tokens=gold_ans, **kwargs
                )
            else:
                predict_results = self.predict(
                    input_texts, **kwargs
                )

        for i in range(len(predict_results)):
            predict_token = predict_results[i]["predict_tokens"][0]
            if ignore_stop_word:
                k = 1
                while predict_token in stop_words\
                        and k < len(predict_results[i]["predict_tokens"]):
                    predict_token = predict_results[i]["predict_tokens"][k]
                    k += 1
            if predict_token.lower() == gold_ans[i].lower():
                p_1 += 1
                predict_results[i]["ans"] = 1
            else:
                predict_results[i]["ans"] = 0

        if len(gold_ans) == 0:
            p_1 = 0
        else:
            p_1 = round(p_1 * 100 / len(gold_ans), 2)

        return predict_results, p_1

    @staticmethod
    def get_objs(samples):
        objs = []
        for sample in samples:
            sub, obj = get_pair(sample)
            objs.append(obj)
        return objs

    def prompt_to_sent_multi_task(self, prompt, sub, mask_len, obj=None):
        assert "[X]" in prompt
        assert "[Y]" in prompt
        sent = prompt.replace("[X]", sub)
        mask_token = self.tokenizer.mask_token
        mask_span = " ".join([mask_token] * mask_len)
        sent = sent.replace("[Y]", mask_span)
        return sent

    # 这么解码很可能前后配不上
    def greedy_search(self, predict_score, mask_index, topk):
        predict_ids = [[] for i in range(topk)]
        predict_probs = [0 for i in range(topk)]
        for index in mask_index:
            score = predict_score[index]
            topk_prob, topk_id = torch.topk(score, topk)
            topk_prob = topk_prob.detach().cpu().numpy()
            topk_id = topk_id.cpu().numpy().tolist()
            for i in range(topk):
                predict_probs[i] += topk_prob[i]
                predict_ids[i].append(topk_id[i])

        predict_results = {
            "predict_tokens": self.tokenizer.batch_decode(predict_ids, skip_special_tokens=True),
            "predict_probs": predict_probs
        }
        return predict_results

    # 这里需要补充
    def iter_decode(self, inputs, mask_index, beam_size, method="order"):
        pass

    def predict_multi_task(self,
                           input_texts: list,
                           batch_size=BATCH_SIZE,
                           topk=10,
                           max_len=256,
                           decode_method="greedy"):
        batch_text = self.partition(input_texts, batch_size)
        predict_results = []
        for idx in tqdm(range(len(batch_text))):
            single_batch_text = batch_text[idx]
            if decode_method == "greedy":
                predict_logits, mask_index = self.get_predict_score(
                    single_batch_text, max_len=max_len
                )
                for i in range(len(single_batch_text)):
                        predict_results.append(self.greedy_search(
                            predict_logits[i], mask_index[i], topk=topk
                        ))
            else:
                raise RuntimeError("no decode method")
        return predict_results

    def evaluate_multi_task(self, relation, samples, mask_len,
                            **kwargs):
        relation_prompt = relation["template"]
        objs = self.get_objs(samples)
        predict_results = [{"predict_tokens": [], "predict_probs": []}
                           for i in range(len(samples))]
        for mask_num in range(1, mask_len + 1):
            input_texts = []
            for sample in samples:
                sub, obj = get_pair(sample)
                sent = self.prompt_to_sent_multi_task(relation_prompt, sub, mask_num)
                input_texts.append(sent)
            results = self.predict_multi_task(
                input_texts, **kwargs
            )
            for i in range(len(samples)):
                predict_results[i]["predict_tokens"].extend(results[i]["predict_tokens"])
                predict_results[i]["predict_probs"].extend(results[i]["predict_probs"])
        for i in range(len(samples)):
            predict_results[i]["predict_probs"], predict_results[i]["predict_tokens"] = \
                sync_sort(predict_results[i]["predict_probs"],
                          predict_results[i]["predict_tokens"])

    def iter_predict(self, relation, samples, decoder, mask_len, batch_size):
        relation_prompt = relation["template"]
        objs = self.get_objs(samples)
        input_sentences = []
        for sample in samples:
            sentence = []
            sub, obj = get_pair(sample)
            for mask_num in range(1, mask_len + 1):
                sent = self.prompt_to_sent_multi_task(relation_prompt, sub, mask_num)
                sentence.append(sent)
            input_sentences.append(sentence)
        predict_results = decoder.decode(input_sentences, batch_size=batch_size)
        return predict_results

    def get_mean_logits(self, mask_pos, mask_index, logits):
        obj_pos = torch.tensor([mask_index[i][mask_pos[i]] for i in range(len(mask_index))]).cuda(self.device)
        obj_pos = torch.reshape(obj_pos, (logits.shape[0], 1, 1))
        index = obj_pos.expand(logits.shape[0], 1, logits.shape[-1])
        predict_logits = torch.gather(logits, 1, index).reshape((logits.shape[0], logits.shape[-1]))
        mean_logits = torch.mean(predict_logits, dim=0)
        return mean_logits

    def eval_sample_with_multi_prompts(
            self, relation_prompts, samples, batch_size, topk=10, max_len=256, ignore_stop_word=True, return_tokens=False
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
            predict_logits, mask_index = self.get_predict_score(
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
                        sub_mentions, size=mentions-1, replace=True
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
            predict_logits, mask_index = self.get_predict_score(
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