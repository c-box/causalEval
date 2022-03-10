from transformers import BartTokenizer, BartForConditionalGeneration
from models.mlm_wrapper import MLMWrapper
from models.gpt_wrapper import GPTWrapper
from utils.utils import load_json_dic, store_vocab2idx, get_pair
import os
from utils.constant import BATCH_SIZE
from tqdm import tqdm


class BartWrapper(MLMWrapper):
    def __init__(self,
                 tokenizer: BartTokenizer,
                 model: BartForConditionalGeneration,
                 vocab2idx_file="data/bart_data/vocab2idx.json",
                 device: int = None):
        super().__init__(tokenizer, model, device=device)
    #     if not os.path.isfile(vocab2idx_file):
    #         store_vocab2idx(tokenizer, vocab2idx_file)
    #     self.vocab2idx = load_json_dic(vocab2idx_file)

    # def token_to_idx(self, token):
    #     return self.vocab2idx[token]
