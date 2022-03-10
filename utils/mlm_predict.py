import torch
from tqdm import tqdm
from utils.constant import CUDA_DEVICE, BATCH_SIZE
from utils.utils import batch_text, score_to_result, get_pair, model_prefix, load_json_dic
from transformers import AutoTokenizer, BertForMaskedLM
import copy
import numpy as np


def find_last(ids, key):
    for i in range(len(ids)-1, -1, -1):
        if ids[i] == key:
            return i
    raise RuntimeError('no mask')


def find_first(ids, key):
    for i in range(len(ids)):
        if ids[i] == key:
            return i
    raise RuntimeError('no mask')


def get_predict_score(input_text, tokenizer, model, num_of_mask=1, args=None, mask_pos=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    if args.max_len is None:
        max_len = 256
    else:
        max_len = args.max_len

    mask_mark = tokenizer.mask_token
    MASK_ID = tokenizer.convert_tokens_to_ids([mask_mark])[0]
    masked_index = []

    inputs = tokenizer.batch_encode_plus(input_text, padding='longest', truncation=True, max_length=max_len)
    input_ids = inputs['input_ids']

    if mask_pos is not None:
        assert len(mask_pos) == len(input_ids)
        for ids, pos in zip(input_ids, mask_pos):
            if pos == 0:
                masked_index.append(find_first(ids, MASK_ID))
            else:
                masked_index.append(find_last(ids, MASK_ID))
    else:
        if num_of_mask == 1:
            for ids in input_ids:
                try:
                    masked_index.append(ids.index(MASK_ID))
                except:
                    print("\n".join(input_text))
                    print(ids)
                    raise RuntimeError("false id")
        elif num_of_mask == -1:
            for ids in input_ids:
                masked_index.append(find_first(ids, MASK_ID))
        else:
            for ids in input_ids:
                masked_index.append(find_last(ids, MASK_ID))

    for key in inputs:
        inputs[key] = torch.tensor(inputs[key]).cuda(cuda_device)
    outputs = model(**inputs)
    prediction_score = outputs[0]
    return prediction_score, masked_index


def mlm_predict(tokenizer, model, input_texts, topk=10, batch_size=BATCH_SIZE,
                obj_tokens=None, num_of_mask=1, args=None, mask_pos=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    if obj_tokens is None:
        batch_input_text = batch_text(input_texts, batch_size=batch_size)
    else:
        batch_input_text, batch_obj_tokens = batch_text(input_texts, batch_size=batch_size, obj_tokens=obj_tokens)

    if mask_pos is not None:
        assert len(input_texts) == len(mask_pos)
        batch_mask_pos = batch_text(mask_pos, batch_size=batch_size)
        assert len(batch_input_text) == len(batch_mask_pos)

    if model_prefix(args.model_name) == "roberta":
        roberta_vocab2idx = load_json_dic("data/roberta_data/vocab2idx.json")
    else:
        roberta_vocab2idx = None

    model = model.cuda(cuda_device)
    model.eval()
    predict_results = []

    for idx in tqdm(range(len(batch_input_text))):
        if obj_tokens is None:
            input_text = batch_input_text[idx]
        else:
            input_text = batch_input_text[idx]
            single_batch_obj = batch_obj_tokens[idx]

        if mask_pos is None:
            single_batch_pos = None
        else:
            single_batch_pos = batch_mask_pos[idx]

        prediction_score, masked_index = \
            get_predict_score(input_text, tokenizer, model,
                              num_of_mask=num_of_mask, args=args, mask_pos=single_batch_pos)
        for i in range(len(masked_index)):
            score = prediction_score[i][masked_index[i]]
            if obj_tokens is None:
                predicted_tokens, predicted_prob = score_to_result(score, topk, tokenizer)
                predict_results.append({'predict_tokens': predicted_tokens,
                                        'predict_prob': predicted_prob})
            else:
                predicted_tokens, predicted_prob, obj_prob, obj_rank, mrr \
                    = score_to_result(score, topk, tokenizer, single_batch_obj[i], roberta_vocab2idx=roberta_vocab2idx)
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


def mlm_prdict_results(tokenizer, model, relation_template, relation_samples, k=10, batch_size=BATCH_SIZE, args=None):
    input_sentences = []
    gold_obj = []
    p_1 = 0
    p_10 = 0
    for sample in relation_samples:
        if 'sub_label' not in sample:
            sample = sample['sample']
        sub = sample['sub_label']
        obj = sample['obj_label']
        gold_obj.append(obj)

        input_sentence = relation_template.replace('[X]', sub)
        input_sentence = input_sentence.replace('[Y]', tokenizer.mask_token)

        input_sentences.append(input_sentence)
    predict_results = mlm_predict(tokenizer, model,
                                  input_texts=input_sentences, topk=k,
                                  obj_tokens=gold_obj, batch_size=batch_size,
                                  args=args)
    for i in range(len(predict_results)):
        predict_tokens = predict_results[i]['predict_tokens']
        predict_prob = predict_results[i]['predict_prob']
        obj = gold_obj[i]
        if obj == predict_tokens[0]:
            p_1 += 1
            predict_results[i]['predict_ans'] = True
        else:
            predict_results[i]['predict_ans'] = False
        if obj in predict_tokens[: 10]:
            p_10 += 1
    p_1 = round(p_1 * 100 / len(gold_obj), 2)
    p_10 = round(p_10 * 100 / len(gold_obj), 2)
    print(relation_template)
    print("P@1: {}, P@10: {}".format(p_1, p_10))
    return predict_results, p_1, p_10


def evaluate_samples(args, relation, relation_samples, tokenizer, model):
    relation_template = relation['template']
    input_sentences = []
    gold_obj = []
    p_1 = 0
    p_10 = 0

    for relation_sample in relation_samples:
        if 'sample' in relation_sample:
            sample = relation_sample['sample']
        else:
            sample = relation_sample
        sub = sample['sub_label']
        obj = sample['obj_label']
        gold_obj.append(obj)

        input_sentence = relation_template.replace('[X]', sub)
        input_sentence = input_sentence.replace('[Y]', tokenizer.mask_token)
        input_sentences.append(input_sentence)

    predict_results = mlm_predict(tokenizer, model,
                                  input_texts=input_sentences, batch_size=args.batch_size, args=args)

    for i in range(len(predict_results)):
        predict_tokens = predict_results[i]['predict_tokens']
        obj = gold_obj[i]
        if obj == predict_tokens[0]:
            p_1 += 1
        if obj in predict_tokens[: 10]:
            p_10 += 1
    p_1 = round(p_1 * 100 / len(gold_obj), 2)
    p_10 = round(p_10 * 100 / len(gold_obj), 2)
    return predict_results, p_1, p_10


def count_sub_words(token, tokenizer):
    text_ids = tokenizer.encode(token, add_special_tokens=False)
    return len(text_ids)


def count_roberta_sub_words(template, obj, tokenizer):
    sub_exclude = template.replace("[Y]", "")
    sub_include = template.replace("[Y]", obj)
    sub_exclude = sub_exclude.replace("  ", " ")
    sub_include = sub_include.replace("  ", " ")
    sub_exclude_len = count_sub_words(sub_exclude, tokenizer)
    sub_include_len = count_sub_words(sub_include, tokenizer)
    return sub_include_len - sub_exclude_len


def template_to_sent(template, sub, obj, tokenizer):
    sent = template.replace('[X]', sub)
    mask_token = tokenizer.mask_token
    if "roberta" in tokenizer.name_or_path:
        mask_cnt = count_roberta_sub_words(sent, obj, tokenizer)
    else:
        mask_cnt = count_sub_words(obj, tokenizer)
    mask_obj = " ".join([mask_token] * mask_cnt)
    sent = sent.replace('[Y]', mask_obj)
    return sent


def get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def get_predict_score_with_multi_mask(tokenizer, model, input_text, args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    mask_mark = tokenizer.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(mask_mark)

    inputs = tokenizer.batch_encode_plus(
        input_text, padding='longest', truncation=True, max_length=256
    )

    input_ids = inputs['input_ids']
    mask_index = []
    for ids in input_ids:
        index = get_index(ids, mask_id)
        mask_index.append(index)

    for key in inputs:
        inputs[key] = torch.tensor(inputs[key]).cuda(cuda_device)
    outputs = model(**inputs)
    prediction_score = outputs[0]
    return prediction_score, mask_index


def score_to_id(score):
    score = torch.softmax(score, dim=-1)
    predict_id = torch.argmax(score).item()
    return predict_id


def beam_search(beam_size, predict_score, mask_index, tokenizer):
    candidates = []
    que = []
    for index in mask_index:
        score = predict_score[index]
        score = torch.softmax(score, dim=-1)
        original_prob, original_index = torch.topk(score, 1000)
        original_prob = original_prob.detach().cpu().numpy()
        original_index = original_index.cpu().numpy().tolist()

        predicted_prob = []
        predicted_index = []

        for idx, prob in zip(original_index, original_prob):
            if "##" in tokenizer.convert_ids_to_tokens(idx):
                predicted_prob.append(prob)
                predicted_index.append(idx)
                if len(predicted_index) == beam_size:
                    break

        if len(candidates) == 0:
            for idx, prob in zip(predicted_index, predicted_prob):
                prob = np.log(prob)
                candidates.append({"idx": [idx], "prob": prob})
        else:
            for idx, prob in zip(predicted_index, predicted_prob):
                prob = np.log(prob)
                for candidate in candidates:
                    cur_idx = candidate['idx']
                    cur_prob = candidate['prob']
                    que.append({"idx": cur_idx + [idx], "prob": prob + cur_prob})
            candidates = copy.copy(que)
            que = []
    que = sorted(candidates, key=lambda x: x['prob'], reverse=True)
    for i in range(len(que)):
        idx = que[i]['idx']
        token = tokenizer.decode(idx)
        if len(token.split(' ')) == 1 and "##" not in token:
            return token
    return None


def greedy_search(predict_score, mask_index, tokenizer):
    ids = []
    for index in mask_index:
        score = predict_score[index]
        predict_id = score_to_id(score)
        ids.append(predict_id)
    predict_token = tokenizer.decode(ids)
    return predict_token


def mlm_predict_with_multi_mask(tokenizer, model,
                                input_texts, batch_size=BATCH_SIZE,
                                decode_method='greedy', beam_size=100,
                                args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE
    batch_input_text = batch_text(input_texts, batch_size=batch_size)

    model = model.cuda(cuda_device)
    model.eval()
    predict_results = []
    for idx in tqdm(range(len(batch_input_text))):
        input_text = batch_input_text[idx]
        prediction_score, mask_index = get_predict_score_with_multi_mask(tokenizer, model, input_text, args=args)
        for i in range(len(mask_index)):
            if decode_method == 'greedy':
                predict_token = greedy_search(prediction_score[i], mask_index[i], tokenizer)
            else:
                raise RuntimeError("no decode method")
            predict_results.append(predict_token)
    return predict_results


def evaluate_samples_with_multi_mask(args, relation, relation_samples, tokenizer, model):
    relation_template = relation['template']
    input_sentences = []
    gold_sub = []
    gold_obj = []
    p_1 = 0
    for sample in relation_samples:
        sub, obj = get_pair(sample)
        gold_sub.append(sub)
        gold_obj.append(obj)
        sent = template_to_sent(relation_template, sub, obj, tokenizer)
        input_sentences.append(sent)

    predict_results = mlm_predict_with_multi_mask(tokenizer, model, input_sentences, batch_size=args.batch_size,
                                                  decode_method=args.decode_method, beam_size=args.beam_size, args=args)

    predict_ans = []
    for i in range(len(predict_results)):
        predict_token = predict_results[i]
        obj = gold_obj[i]
        if obj in predict_token.split():
            p_1 += 1
            predict_ans.append(True)
        else:
            predict_ans.append(False)
    if len(gold_obj) == 0:
        p_1 = 0
    else:
        p_1 = round(p_1 * 100 / len(gold_obj), 2)

    return predict_results, p_1


def greedy_search_subtokens(predict_score, mask_index, tokenizer):
    ids = []
    for index in mask_index:
        score = predict_score[index]
        predict_id = score_to_id(score)
        ids.append(predict_id)
    predict_token = tokenizer.convert_ids_to_tokens(ids)
    return predict_token


def mlm_predict_with_multi_mask_any_sub_word(tokenizer, model,
                                             input_texts, batch_size=BATCH_SIZE,
                                             decode_method='greedy', beam_size=100,
                                             args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE
    batch_input_text = batch_text(input_texts, batch_size=batch_size)

    model = model.cuda(cuda_device)
    model.eval()
    predict_results = []
    for idx in tqdm(range(len(batch_input_text))):
        input_text = batch_input_text[idx]
        prediction_score, mask_index = get_predict_score_with_multi_mask(tokenizer, model, input_text, args=args)
        for i in range(len(mask_index)):
            predict_token = greedy_search_subtokens(prediction_score[i], mask_index[i], tokenizer)
            predict_results.append(predict_token)
    return predict_results


def evaluate_samples_with_multi_mask_any_subword(args, relation, relation_samples, tokenizer, model):
    relation_template = relation['template']
    input_sentences = []
    gold_sub = []
    gold_obj = []
    p_1 = 0
    for sample in relation_samples:
        if 'sample' in sample:
            sample = sample['sample']
        sub = sample['sub_label']
        obj = sample['obj_label']
        gold_sub.append(sub)
        gold_obj.append(obj)
        sent = template_to_sent(relation_template, sub, obj, tokenizer)
        input_sentences.append(sent)

    predict_results = mlm_predict_with_multi_mask_any_sub_word(
        tokenizer, model, input_sentences, batch_size=args.batch_size,
        decode_method=args.decode_method, beam_size=args.beam_size, args=args
    )

    predict_ans = []
    for i in range(len(predict_results)):
        predict_token = predict_results[i]
        obj = gold_obj[i]
        obj_tokens = tokenizer.tokenize(obj)
        flag = False
        for obj_sub in obj_tokens:
            for predict_sub in predict_token:
                if obj_sub == predict_sub:
                    flag = True
                    break
        if flag:
            p_1 += 1
            predict_ans.append(True)
        else:
            predict_ans.append(False)
    if len(gold_obj) == 0:
        p_1 = 0
    else:
        p_1 = round(p_1 * 100 / len(gold_obj), 2)

    return predict_results, p_1
