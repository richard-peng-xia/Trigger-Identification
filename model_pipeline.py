import torch
import torch.nn as nn
from torch.nn import functional as F


class Pipeline(nn.Module):
    def __init__(self, Encoder, Interact, Integrate, args):
        super().__init__()
        # model layers
        self.encoder = Encoder(args)
        self.interaction = Interact(args, input_size = self.encoder.sent_size)
        self.integration = Integrate(args, input_size = self.interaction.node_size)
        self.activation = nn.LeakyReLU(0.1)
        self.trigger_classifier = nn.Linear(self.interaction.node_size, 4)
        self.verify_classifier = nn.Linear(self.interaction.node_size, 3)
        # loss function
        self.loss_func = F.cross_entropy

    def forward(self, batch_data):
        fields = ['mid', 'cid', 'pid', 'time', 'content_clean', 'trigger', 'verify']
        mids, cids, pids, time, mes, yt, yv = tuple(
            [batch_data[field] for field in fields])
        structure = (mids, cids, pids, time)
        mes_embed = self.encoder(mes)
        # update mesence representation using propagation information
        mes_update = self.interaction(mes_embed, structure)
        # classify mesences to identify trigger
        yt_pred = self.trigger_classifier(self.activation(mes_update))
        lt = self.loss_func(yt_pred, yt, weight=None)
        # integrate mesences to form cascade representation
        cascade, yv_cas = self.integration(
            mes_update, yv, structure, yt_pred=yt_pred)
        yv_pred = self.verify_classifier(self.activation(cascade))
        lv = self.loss_func(yv_pred, yv_cas, weight=None)
        # update
        output = {
            'mid': mids,
            'cid': [cids[i] for i in range(len(cids)) if pids[i] == 'None'],
            'trigger_preds': yt_pred,
            'trigger_targets': yt,
            'trigger_loss': lt,
            'verify_preds': yv_pred,
            'verify_targets': yv_cas,
            'verify_loss': lv,
        }
        return output

# pipeline = Pipeline(Encoder, Interact, Integrate, args)
# pipeline(batch_data1)