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


def test_model_expression(envir, module_transfo, input_equation, params, verbose=True, latex_form=False):
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

            if matches or error == -1 and first_valid_num is None:
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

    if verbose:
        if first_valid_num is None:
            print('Could not solve')
        else:
            print('Solved in beam search')
        print("")
        print("")

    return first_valid_num


if __name__ == '__main__':

    # Input equation to simplify
    input_eq = '(-ab(1, 2)*ab(1, 3)*ab(2, 4)*ab(2, 5)*sb(2, 4)*sb(3, 5) - ab(1, 2)*ab(1, 3)*ab(2, 4)*ab(3, 5)*sb(3, 4)*sb(3, 5) + ab(1, 3)**2*ab(2, 4)*ab(2, 5)*sb(3, 4)*sb(3, 5) - ab(1, 3)*ab(1, 4)*ab(2, 3)*ab(2, 5)*sb(3, 4)*sb(3, 5))/(ab(1, 5)*ab(2, 3)*ab(3, 4)*ab(4, 5)**2*sb(1, 2)*sb(4, 5))'

    parameters = AttrDict({
        # Experiment Name
        'exp_name': 'Test_eval_spin_hel',

        # Specify the dump path
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/scratch/',
        'exp_id': 'test',
        'save_periodic': 0,
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
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5_c/checkpoint.pth',
        # 'reload_model': '',

        # Trainer param
        'export_data': False,

        # Data path (not needed)
        'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid',
        #'reload_data': '',
        'reload_size': '',
        'epoch_size': 1000,
        'max_epoch': 500,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': 'valid_func_simple_acc',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 1,

        # Evaluation
        'eval_only': True,
        'numerical_check': True,
        'eval_verbose': 2,
        'eval_verbose_print': True,
        'beam_eval': True,
        'beam_size': 30,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': False,
        'nucleus_p': 0.95,
        'temperature': 1,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,

        # Specify the path to Spinors Mathematica Library
        'lib_path': '/Users/aurelien/Documents/Package_lib/Spinors-1.0',
        'mma_path': None,
    })

    check_model_params(parameters)

    # Start the logger
    init_distributed_mode(parameters)
    logger = initialize_exp(parameters)
    init_signal_handler()

    environment.utils.CUDA = not parameters.cpu

    # Load the model and environment
    env = build_env(parameters)

    modules = build_modules(env, parameters)

    # start the wolfram session
    if parameters.numerical_check:
        session = initialize_numerical_check(parameters.npt_list[0], lib_path=parameters.lib_path)
        env.session = session

    first_num = test_model_expression(env, modules, input_eq, parameters, verbose=True, latex_form=True)
    print(first_num)

    if parameters.numerical_check:
        env.session.stop()

