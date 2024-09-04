import torch.nn as nn

class MultiLabelLoss(nn.Module):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        return self.ce_loss(outputs, targets)