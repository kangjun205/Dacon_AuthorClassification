import torch
from torch.utils.data import DataLoader
from .datasets import TextDataset

def get_dataloader(texts, labels, tokenizer, max_len, batch_size, shuffle=True):
    """_summary_

    Args:
        texts (list[list[str]]): cleaned text
        labels (list[int]): target label
        tokenizer (callable[[str], list[str]]): tokenizer
        max_len (int): max vocabulary length
        batch_size (int): mini batch size
        shuffle (bool, optional): shuffle the dataset

    Returns:
        _type_: _description_
    """    
    dataset = TextDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_dataloaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer, max_len, batch_size):
    """_summary_

    Args:
        train_texts (_type_): _description_
        train_labels (_type_): _description_
        val_texts (_type_): _description_
        val_labels (_type_): _description_
        test_texts (_type_): _description_
        test_labels (_type_): _description_
        tokenizer (_type_): _description_
        max_len (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_dataloader = get_dataloader(train_texts, train_labels, tokenizer, max_len, batch_size)
    val_dataloader = get_dataloader(val_texts, val_labels, tokenizer, max_len, batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_texts, test_labels, tokenizer, max_len, batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader