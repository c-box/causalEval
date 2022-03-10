from models.mlm_wrapper import MLMWrapper
from transformers import RobertaTokenizer, RobertaForMaskedLM
from utils.utils import load_json_dic, store_json_dic, store_vocab2idx
import os


class RobertaWrapper(MLMWrapper):
    def __init__(self,
                 tokenizer: RobertaTokenizer,
                 model: RobertaForMaskedLM,
                 vocab2idx_file="data/roberta_data/vocab2idx.json",
                 device: int = None):
        super().__init__(tokenizer, model, device=device)
        # if not os.path.isfile(vocab2idx_file):
        #     store_vocab2idx(tokenizer, vocab2idx_file)
        # self.vocab2idx = load_json_dic(vocab2idx_file)
        # self.vocab = self.vocab2idx

    def token_to_idx(self, token):
        return self.vocab2idx[token]
