from transformers import BertTokenizer
from transformers import AutoTokenizer


def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def get_tokenizer_deberta():
    return AutoTokenizer.from_pretrained("mrm8488/deberta-v3-small-finetuned-cola")

def get_vocab_size(tokenizer):
    return len(tokenizer.vocab)

