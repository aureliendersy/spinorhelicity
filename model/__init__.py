# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""

"""


from logging import getLogger
import os
import torch

from .transformer import TransformerModel


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # reload a pretrained model
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules['encoder'] = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
    modules['decoder'] = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)

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


# Download links to the trained models
MODULE_REGISTRAR = {'4-pt': 'https://drive.google.com/uc?export=download&id=1lq46Hc_eF8khsoC4k-lsMjZpaY0Pco29',
                    '5-pt': 'https://drive.google.com/uc?export=download&id=1hSyRKcsMjxgVZ3DB1EkwD0OFHxoId4FZ',
                    '6-pt': 'https://drive.google.com/uc?export=download&id=1JFhH-6UPXvLdFb3Gd-DFhCS7hGmgXQJw',
                    '4-pt-contrastive':'https://drive.google.com/uc?export=download&id=1jG3dwWmsy2exiBV9ip4p7Fy-KyA0-raz',
                    '5-pt-contrastive': 'https://drive.google.com/uc?export=download&id=1zl_T-dPjqbkVn8677Oqlw-Pfufv8RMKC',
                    '6-pt-contrastive': 'https://drive.google.com/uc?export=download&id=1uFzMdGRNeq_WA9oLVBMGJ9ajIcjETziA'}