from utils.utils import MODEL_PATH, model_prefix
from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    GPT2LMHeadModel, GPT2Tokenizer, \
    RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, BertForMaskedLM, \
    BartForConditionalGeneration, BartTokenizer
from models.bert_wrapper import BertWrapper
from models.roberta_wrapper import RobertaWrapper
from models.gpt_wrapper import GPTWrapper
from models.bart_wrapper import BartWrapper


def build_model_wrapper(model_name, device=None, args=None, model_path=None):
    if model_path is None:
        if model_name in MODEL_PATH:
            model_path = MODEL_PATH[model_name]
        else:
            model_path = model_name
            # raise RuntimeError('model not exsit')
    if model_prefix(model_name) == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForMaskedLM.from_pretrained(model_path)
        model_wrapper = RobertaWrapper(tokenizer, model, device=device)
    elif model_prefix(model_name) == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(model_path)
        model_wrapper = BertWrapper(tokenizer, model, device=device)
    elif model_prefix(model_name) == "gpt2":
        assert args is not None
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model_wrapper = GPTWrapper(tokenizer, model, args.gpt_method, args.generate_len, device=device)
    elif model_prefix(model_name) == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model_wrapper = BartWrapper(tokenizer, model, device=device)
    else:
        raise RuntimeError("wrong model name")
    return model_wrapper
