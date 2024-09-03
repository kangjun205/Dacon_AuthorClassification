from transformers import BertTokenizer

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

def get_vocab_size(tokenizer):
    return len(tokenizer.vocab)