import torch.nn as nn
from transformers import BertModel
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class DebertaForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(DebertaForMultiLabelClassification, self).__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            "mrm8488/deberta-v3-small-finetuned-cola", 
            num_labels=2 
        )

        self.deberta.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class XLMRobertaForMultiClassClassification(nn.Module):
    def __init__(self, num_labels):
        super(XLMRobertaForMultiClassClassification, self).__init__()
        self.roberta = AutoModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = self.classifier(cls_output)
        return cls_output