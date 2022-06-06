import torch.nn as nn


class Interact(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.node_size = input_size

    def forward(self, mes_embed, structure):
        return mes_embed
