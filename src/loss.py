import torch.nn as nn

class MultiLabelLoss(nn.Module):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets):
        return self.bce_loss(outputs, targets)