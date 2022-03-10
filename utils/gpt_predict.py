from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from utils.utils import batch_text, score_to_result, get_pair, MODEL_PATH
from utils.constant import CUDA_DEVICE
from tqdm import tqdm
import torch


def get_predict_score(input_text, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel,
                      args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.batch_encode_plus(input_text, padding='longest',
                                         truncation=True, max_length=256,
                                         return_tensors="pt", add_special_tokens=True)
    inputs = inputs.to(cuda_device)
    outputs = model(**inputs, return_dict=True)
    prediction_score = outputs.logits
    return prediction_score


def score_to_id(score):
    score = torch.softmax(score, dim=-1)
    predict_id = torch.argmax(score).item()
    return predict_id


def greedy_search(predict_score, tokenizer: GPT2Tokenizer):
    ids = []
    for index in range(len(predict_score)):
        score = predict_score[index]
        predict_id = score_to_id(score)
        ids.append(predict_id)
    predict_token = tokenizer.convert_ids_to_tokens(ids)
    return predict_token


def gpt_next_token(tokenizer: GPT2Tokenizer, model,
                   input_texts, topk=10, batch_size=8, args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    batch_input_text = batch_text(input_texts, batch_size=batch_size)
    model = model.cuda(cuda_device)
    predict_results = []
    for input_text in tqdm(batch_input_text):
        prediction_score = get_predict_score(input_text, tokenizer, model, args=args)
        for i in range(len(prediction_score)):
            text = input_text[i]
            prompt_length = len(tokenizer.encode(text, add_special_tokens=False))

            score = prediction_score[i][prompt_length-1]
            predicted_tokens, predicted_prob = score_to_result(score, topk, tokenizer)
            predict_results.append({'predict_tokens': predicted_tokens,
                                    'predict_prob': predicted_prob})
    return predict_results


def gpt_generate(input_text, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, args=None, max_len=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE

    inputs = tokenizer.batch_encode_plus(
        input_text, padding='longest',
        truncation=True, max_length=256,
        return_tensors="pt"
    )
    inputs = inputs.to(cuda_device)
    # 参数可修改
    outputs = model.generate(
        **inputs, do_sample=False, max_length=max_len, pad_token_id=tokenizer.eos_token_id
    )
    return outputs


def gpt_generate_sequence(tokenizer: GPT2Tokenizer, model, input_texts, batch_size=8, args=None):
    if args is not None:
        cuda_device = args.cuda_device
    else:
        cuda_device = CUDA_DEVICE
    batch_input_text = batch_text(input_texts, batch_size=batch_size)
    model = model.cuda(cuda_device)
    model.eval()
    tokenizer.padding_side = "left"

    predict_results = []

    max_len_str = max(input_texts, key=len, default="")
    max_len = len(tokenizer.encode(max_len_str)) + 8

    for input_text in tqdm(batch_input_text):
        outputs = gpt_generate(input_text, tokenizer, model, args, max_len=max_len)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(len(outputs)):
            prompt_length = len(input_text[i].split())
            # out = outputs[i]
            # predict_tokens = tokenizer.decode(out, skip_special_tokens=True).split()
            predict_tokens = outputs[i].split()
            predict_token = predict_tokens[prompt_length]
            predict_seq = predict_tokens[prompt_length:]
            predict_results.append({"predict_tokens": [predict_token], "predict_seq": predict_seq})
    return predict_results


def gpt_pipline(input_texts, args, batch_size=8):
    cuda_device = args.cuda_device
    batch_input_text = batch_text(input_texts, batch_size=batch_size)

    model_path = MODEL_PATH[args.model_name]
    generator = pipeline("text-generation", model=model_path, device=cuda_device)
    # generator.tokenizer.pad_token = generator.tokenizer.eos_token

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    max_len_str = max(input_texts, key=len, default="")
    max_len = len(tokenizer.encode(max_len_str)) + 8

    predict_results = []
    for input_text in tqdm(batch_input_text):
        outputs = generator(
            input_text, max_length=max_len, num_return_sequences=1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        for i in range(len(input_text)):
            predict_seq = outputs[i][0]["generated_text"].split()
            prompt_length = len(input_text[i].split())
            predict_token = predict_seq[prompt_length]
            predict_seq = predict_seq[prompt_length:]
            predict_results.append({"predict_tokens": [predict_token],
                                    "predict_seq": predict_seq})

    return predict_results


def evaluate_samples_using_gpt(args, relation, relation_samples, tokenizer, model):
    relation_template = relation['template']
    input_sentences = []
    gold_obj = []
    p_1 = 0

    for sample in relation_samples:
        sub, obj = get_pair(sample)
        gold_obj.append(obj)
        sent = relation_template.replace("[X]", sub)
        sent = sent.replace("[Y]", "").strip(".").strip(" ")
        input_sentences.append(sent)

    if args.gpt_method == "next_token":
        predict_results = gpt_next_token(tokenizer, model, input_sentences,
                                         batch_size=args.batch_size, args=args)
    elif args.gpt_method == "generate_sequence":
        predict_results = gpt_generate_sequence(tokenizer, model, input_sentences,
                                                batch_size=args.batch_size, args=args)
    elif args.gpt_method == "generate_pipline":
        predict_results = gpt_pipline(input_sentences, args, batch_size=args.batch_size)
    else:
        raise RuntimeError("wrong methods")

    for i in range(len(predict_results)):
        if args.generate_len > 1:
            assert args.gpt_method == "generate_sequence"
            predict_tokens = " ".join(predict_results[i]["predict_seq"][: args.generate_len])
        else:
            predict_tokens = predict_results[i]["predict_tokens"][0]
        obj = gold_obj[i]
        if obj in predict_tokens:
            p_1 += 1
    if len(gold_obj) == 0:
        p_1 = 0
    else:
        p_1 = round(p_1 * 100 / len(gold_obj), 2)

    return predict_results, p_1
