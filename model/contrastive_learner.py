"""
File containing the necessary modules for creating and training a Transformer Encoder used for contrastive learning
"""

import torch
import torch.nn as nn
from model.transformer import TransformerModel
from collections import OrderedDict
from logging import getLogger

logger = getLogger()


class FFNHead(nn.Module):

    def __init__(self, embed_dim, layer_nums):
        super().__init__()
        self.activation = nn.ReLU()
        self.embed_dim = embed_dim
        self.layer_nums = layer_nums

        layer_list = []

        for i in range(self.layer_nums):
            layer_list.append(('layer_%d' % (i + 1), nn.Linear(self.embed_dim, self.embed_dim)))

            if i + 1 < self.layer_nums:
                layer_list.append(('activation_%d' % (i + 1), self.activation))

        self.model = nn.Sequential(OrderedDict(layer_list))

    def forward(self, inputx):
        x = self.model(inputx)
        return x


class TransformerEncoderC(TransformerModel):

    def __init__(self, params, id2word):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__(params, id2word, True, False)
        self.ffnhead = FFNHead(params.emb_dim, params.head_layers)

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            raise Exception("Not supposed to run in predict mode: %s")
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, cache=None, previous_state=None):

        # Pass through the Encoder and get (Num words, Bs, Embed Dim)
        tensor = TransformerModel.fwd(self, x, lengths, causal, src_enc, src_len, positions, cache, previous_state)

        # Average over the word representations get (Bs, Embed Dim)
        tensor = torch.mean(tensor, 0)

        # Pass through the final layer
        tensor = self.ffnhead(tensor)

        return tensor


def build_modules_contrastive(env, params):
    """
    Build modules.
    """
    modules = {'encoder_c': TransformerEncoderC(params, env.id2word)}

    # reload pretrained modules
    if params.reload_model != '':
        logger.info(f"Reloading modules from {params.reload_model} ...")
        if not params.cpu:
            reloaded = torch.load(params.reload_model)
        else:
            reloaded = torch.load(params.reload_model, map_location=torch.device('cpu'))
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith('module.') for k2 in reloaded[k].keys()]):
                reloaded[k] = {k2[len('module.'):]: v2 for k2, v2 in reloaded[k].items()}
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}")

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
