"""
The test desired model on a given input expression
"""


from statistics import mean
import torch
from add_ons.slurm import init_signal_handler, init_distributed_mode
from environment.utils import AttrDict, to_cuda, initialize_exp, convert_sp_forms, reorder_expr
from environment import build_env
import environment
from model import build_modules, check_model_params
from add_ons.mathematica_utils import *
import sympy as sp
from sympy import latex
import gdown
import os, csv
import user_args as args


def test_model_expression(envir, module_transfo, input_equation, params, verbose=True, latex_form=False, dir_out=None):
    """
    Test the capacity of the transformer model to resolve a given input
    :param envir:
    :param module_transfo:
    :param input_equation:
    :param params:
    :param verbose:
    :param latex_form:
    :return:
    """

    # load the transformer
    encoder = module_transfo['encoder']
    decoder = module_transfo['decoder']
    encoder.eval()
    decoder.eval()

    f = sp.parse_expr(input_equation, local_dict=envir.func_dict)
    if params.canonical_form:
        f = reorder_expr(f)
    f = f.cancel()
    f_prefix = envir.sympy_to_prefix(f)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in x1_prefix] + [envir.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_size = params.beam_size
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_size,
                                           length_penalty=params.beam_length_penalty,
                                           early_stopping=params.beam_early_stopping,
                                           max_len=params.max_len,
                                           stochastic=params.nucleus_sampling,
                                           nucl_p=params.nucleus_p,
                                           temperature=params.temperature)
        assert len(beam) == 1
    hypotheses = beam[0].hyp
    assert len(hypotheses) == beam_size

    if verbose:
        print(f"Input function f: {f}")
        if latex_form:
            print(latex(f))
        print("")

    first_valid_num = None

    data_out = []

    # Print out the scores and the hypotheses
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [envir.id2word[wid] for wid in ids]  # convert to prefix
        hyp_disp = ''
        # Parse the identities if required
        try:
            hyp = envir.prefix_to_infix(tok)
            if '&' in tok:
                prefix_info = tok[tok.index('&'):]
                info_infix = envir.scr_prefix_to_infix(prefix_info)
            else:
                info_infix = ''

            # convert to infix
            hyp = envir.infix_to_sympy(hyp)  # convert to SymPy
            hyp_disp = convert_sp_forms(hyp, env.func_dict)

            # When integrating the symbol
            if params.numerical_check:
                hyp_mma = sp_to_mma(hyp, envir.npt_list, params.bracket_tokens, envir.func_dict)
                f_sp = envir.infix_to_sympy(envir.prefix_to_infix(envir.sympy_to_prefix(f)))
                tgt_mma = sp_to_mma(f_sp, envir.npt_list, params.bracket_tokens, envir.func_dict)
                matches, error = check_numerical_equiv(envir.session, hyp_mma, tgt_mma)
            else:
                matches = None
                error = None

            res = "Unknown" if matches is None else ("OK" if (matches or error == -1) else "NO")
            # remain = "" if matches else " | {} remaining".format(remaining_diff)
            remain = ""

            if (matches or error == -1) and first_valid_num is None:
                first_valid_num = num + 1

        except:
            res = "INVALID PREFIX EXPRESSION"
            hyp = tok
            remain = ""
            info_infix = ''

        if verbose:
            # print result
            if latex_form:
                print("%.5f  %s %s %s %s" % (score, res, hyp, info_infix, latex(hyp_disp)))
            else:
                print("%.5f  %s %s  %s %s" % (score, res, hyp, info_infix, remain))

        if res == "INVALID PREFIX EXPRESSION":
            data_out.append(['INVALID EXPR', 'NO', score, ''])
        else:
            hyp_mma = sp_to_mma(hyp, envir.npt_list, params.bracket_tokens, envir.func_dict)
            data_out.append([hyp, res, score, hyp_mma])

    if verbose:
        if first_valid_num is None:
            print('Could not solve')
        else:
            print('Solved in beam search')
        print("")
        print("")

    if dir_out is not None:
        f_sp = envir.infix_to_sympy(envir.prefix_to_infix(envir.sympy_to_prefix(f)))
        input_mma = sp_to_mma(f_sp, envir.npt_list, params.bracket_tokens, envir.func_dict)

        file_path_out = os.path.join(dir_out, 'test_model.csv')
        header_out = ['Guess', 'Valid', 'Score', 'MMA_form']
        first_line = [f, 'INPUT', 0.0, input_mma]
        data_out.insert(0, first_line)

        with open(file_path_out, 'w', encoding='UTF8', newline='') as fout:
            writer = csv.writer(fout)

            # write the header
            writer.writerow(header_out)

            # write multiple rows
            writer.writerows(data_out)

    return first_valid_num


if __name__ == '__main__':

    # Whether to load parameters from file
    user_run = True

    path_model1 = os.path.join('model/trained_models/', 'simplifier_model.pth')

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

    else:
        path_mod1 = '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5_c/checkpoint.pth'

    # Input equation to simplify
    input_eq = args.input_eq

    parameters = AttrDict({
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 3,
        'max_scrambles': 3,
        'save_info_scr': False,
        'save_info_scaling': False,
        'numeral_decomp': True,
        'int_base': 10,
        'max_len': 2048,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': False,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'n_max_positions': 2560,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,

        # Specify the path to the simplifier model
        'reload_model': path_mod1,

        # Evaluation
        'beam_eval': True,
        'beam_size': args.beam_size,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': args.nucleus_sampling,
        'nucleus_p': args.nucleus_p,
        'temperature': args.temperature,

        # SLURM/GPU param
        'cpu': True,

        # Specify the path to Spinors Mathematica Library
        'lib_path': args.spinors_lib_path,
        'mma_path': args.mathematica_path,
        'numerical_check': True,
    })

    # Start the logger
    check_model_params(parameters)

    environment.utils.CUDA = not parameters.cpu

    # Load the model and environment
    env = build_env(parameters)

    modules = build_modules(env, parameters)

    # start the wolfram session
    if parameters.numerical_check:
        session = initialize_numerical_check(parameters.npt_list[0], lib_path=parameters.lib_path)
        env.session = session

    first_num = test_model_expression(env, modules, input_eq, parameters, verbose=True, latex_form=True,
                                      dir_out=args.dir_out)
    print("First solution at position {}".format(first_num))

    if parameters.numerical_check:
        env.session.stop()
