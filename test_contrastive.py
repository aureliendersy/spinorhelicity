"""
Module for testing a trained model for contrastive learning
"""
import sympy as sp
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from environment.utils import AttrDict, reorder_expr, to_cuda
from add_ons.slurm import init_signal_handler, init_distributed_mode
import environment
import os, gdown
import user_args as args
from environment.utils import initialize_exp, get_numerator_lg_scaling
from environment import build_env
from model.contrastive_learner import build_modules_contrastive
from add_ons.mathematica_utils import initialize_numerical_check
from model.contrastive_simplifier import total_simplification, blind_constants


def test_expression_factors(envir, module_transfo, input_equation, params, factor_mask=False, const_blind=True):

    # load the transformer encoder
    encoder_c = module_transfo['encoder_c']
    encoder_c.eval()

    f = sp.parse_expr(input_equation, local_dict=envir.func_dict)
    if params.canonical_form:
        f = reorder_expr(f)
    f = f.cancel()
    if const_blind:
        f, _ = blind_constants(f)
    numerator, _ = sp.fraction(f)

    scales = get_numerator_lg_scaling(numerator, list(envir.func_dict.values()))

    if isinstance(numerator, sp.Add):
        terms = numerator.args
    else:
        terms = [numerator]

    # Similarity
    cossim = nn.CosineSimilarity(dim=-1)

    if not factor_mask:
        t_prefix = [envir.sympy_to_prefix(term) for term in terms]
        t_in = [to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_pre] + [envir.eos_index]).view(-1, 1))[0]
                for t_pre in t_prefix]

        len_in = [to_cuda(torch.LongTensor([len(term)]))[0] for term in t_in]

        # Forward
        with torch.no_grad():

            encoded_terms = [encoder_c('fwd', x=t, lengths=len_in[i], causal=False) for i, t in enumerate(t_in)]

            # Similarity
            similarity_mat = cossim(torch.stack(encoded_terms), torch.transpose(torch.stack(encoded_terms), 0, 1))
    else:
        similarity_mat = torch.zeros(len(terms), len(terms))
        for i, term in enumerate(terms):
            for j, term2 in enumerate(terms[i+1:]):
                newterm, newterm2 = sp.fraction(sp.cancel(term/term2))

                t_prefix1, t_prefix2 = envir.sympy_to_prefix(newterm), envir.sympy_to_prefix(newterm2)
                t_in1, t_in2 = to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_prefix1] + [envir.eos_index]).view(-1, 1))[0],\
                               to_cuda(torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in t_prefix2] + [envir.eos_index]).view(-1, 1))[0]

                len_in1, len_in2 = to_cuda(torch.LongTensor([len(t_in1)]))[0],\
                                   to_cuda(torch.LongTensor([len(t_in2)]))[0]

                # Forward
                with torch.no_grad():
                    encoded1, encoded2 = encoder_c('fwd', x=t_in1, lengths=len_in1, causal=False),\
                                         encoder_c('fwd', x=t_in2, lengths=len_in2, causal=False)

                    similarity_mat[i, i+j+1] = cossim(encoded1, encoded2)
                    similarity_mat[i+j+1, i] = cossim(encoded1, encoded2)
            similarity_mat[i, i] = 1
    return similarity_mat.numpy(), terms


if __name__ == '__main__':

    # Whether to load parameters from file
    user_run = True

    path_model1 = os.path.join('model/trained_models/', 'simplifier_model.pth')
    path_model2 = os.path.join('model/trained_models/', 'contrastive_group_model.pth')

    # Download the models if necessary
    if user_run is True:
        if args.model_path_simplifier is None and not os.path.isfile(path_model1):
            print('Starting download of the simplifier model')
            gdown.download(args.download_path_simplifier, path_model1, quiet=False)
            print('Model downloaded at {}'.format(path_model1))
            path_mod1 = path_model1
        else:
            path_mod1 = args.model_path_simplifier if args.model_path_simplifier is not None else path_model1
            print('Using simplifier model from {}'.format(path_mod1))

        if args.model_path_contrastive is None and not os.path.isfile(path_model2):
            print('Starting download of the contrastive model')
            gdown.download(args.download_path_contrastive, path_model2, quiet=False)
            print('Model downloaded at {}'.format(path_model2))
            path_mod2 = path_model2
        else:
            path_mod2 = args.model_path_contrastive if args.model_path_contrastive is not None else path_model2
            print('Using contrastive model from {}'.format(path_mod2))
    else:
        path_mod1 = '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5_final/checkpoint.pth'
        path_mod2 = '/Users/aurelien/PycharmProjects/spinorhelicity/experiments_contrastive/npt5_contrastive_final/exp2/checkpoint.pth'

    # Input equation to simplify
    input_eq = args.input_eq

    parameters_c_dict = {
        # Name
        'exp_name': 'Simple_simplfication',
        'dump_path': 'experiments',
        'exp_id': 'test',
        'tasks': 'contrastive',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': True,
        'save_info_scaling': True,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 2048,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': True,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'n_heads': 8,
        'dropout': 0,
        'head_layers': 2,
        'n_max_positions': 384,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,

        # Specify the path to the contrastive model
        'reload_model': path_mod2,

        # SLURM/GPU param
        'cpu': True,
        'numerical_check': True,
        'mma_path': args.mathematica_path

    }

    parameters_c = AttrDict(parameters_c_dict)

    parameters_s_dict = deepcopy(parameters_c_dict)
    parameters_s = AttrDict(parameters_s_dict)
    parameters_s.tasks = 'spin_hel'
    parameters_s.max_terms = 3
    parameters_s.max_scrambles = 3
    parameters_s.reduced_voc = False
    parameters_s.save_info_scr = False
    parameters_s.save_info_scaling = False

    parameters_s.n_enc_layers = 3
    parameters_s.n_dec_layers = 3
    parameters_s.n_max_positions = 2560

    # Specify the path to the simplfiier model
    parameters_s.reload_model = path_mod1
    parameters_s.lib_path = args.spinors_lib_path

    parameters_s.beam_size = args.beam_size
    parameters_s.beam_length_penalty = 1
    parameters_s.beam_early_stopping = True
    parameters_s.max_len = 2048
    parameters_s.nucleus_p = args.nucleus_p
    parameters_s.temperature = args.temperature

    environment.utils.CUDA = not parameters_c.cpu

    logger = initialize_exp(parameters_c)

    # Load the model and environment
    env_c = build_env(parameters_c)
    env_s = build_env(parameters_s)
    envs = env_c, env_s
    params = parameters_c, parameters_s

    # Start the wolfram session
    session = initialize_numerical_check(parameters_s.npt_list[0], lib_path=parameters_s.lib_path)
    env_s.session = session
    rng_np = np.random.default_rng(323)
    rng_torch = torch.Generator(device='cuda' if not parameters_c.cpu else 'cpu')
    rng_torch.manual_seed(323)

    simplified_eq = total_simplification(envs, params, input_eq, (rng_np, rng_torch), const_blind=True,
                                         init_cutoff=args.init_cutoff, power_decay=args.power_decay,
                                         dir_out=args.dir_out)

    print('Done')

    env_s.session.stop()
    exit()
