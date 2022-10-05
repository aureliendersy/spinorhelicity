"""
The test desired model on a given input expression
"""


from statistics import mean
import torch
from add_ons.slurm import init_signal_handler, init_distributed_mode
from environment.utils import AttrDict, to_cuda, initialize_exp
from environment import build_env
import environment
from model import build_modules, check_model_params
from add_ons.mathematica_utils import *
import sympy as sp
from sympy import latex


def test_model_expression(environment, module_transfo, input_equation, verbose=True, latex_form=False):
    """
    Test the capacity of the transformer model to resolve a given input
    :param environment:
    :param module_transfo:
    :param input_equation:
    :param verbose:
    :param latex_form:
    :return:
    """

    # load the transformer
    encoder = module_transfo['encoder']
    decoder = module_transfo['decoder']
    encoder.eval()
    decoder.eval()

    f = sp.parse_expr(input_equation, local_dict=environment.func_dict)
    f = f.cancel()
    f_prefix = environment.sympy_to_prefix(f)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([environment.eos_index] + [environment.word2id[w] for w in x1_prefix] + [environment.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # cuda
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_size = params.beam_size
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_size, length_penalty=1,
                                           early_stopping=1, max_len=params.max_len)
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
        tok = [environment.id2word[wid] for wid in ids]  # convert to prefix

        try:
            hyp = environment.prefix_to_infix(tok)  # convert to infix
            hyp = environment.infix_to_sympy(hyp)  # convert to SymPy

            # When integrating the symbol
            if params.numerical_check:
                hyp_mma = sp_to_mma(hyp)
                tgt_mma = sp_to_mma(f)
                matches = check_numerical_equiv(environment.session, hyp_mma, tgt_mma)
            else:
                matches = None

            res = "Unknown" if matches is None else ("OK" if matches else "NO")
            # remain = "" if matches else " | {} remaining".format(remaining_diff)
            remain = ""

            if matches and first_valid_num is None:
                first_valid_num = num + 1

        except:
            res = "INVALID PREFIX EXPRESSION"
            hyp = tok
            remain = ""

        if verbose:
            # print result
            if latex_form:
                print("%.5f  %s  %s %s" % (score, res, hyp, latex(hyp)))
            else:
                print("%.5f  %s  %s %s" % (score, res, hyp, remain))


    if verbose:
        if first_valid_num is None:
            print('Could not solve')
        else:
            print('Solved in beam search')
        print("")
        print("")

    return first_valid_num


if __name__ == '__main__':

    # Example with integrating the symbol (don't need to precise the target, we can check it with MMA)
    input_eq = '(-ab(p1, p2)*ab(p1, p3)**2*ab(p3, p4) - ab(p1, p3)**2*ab(p1, p4)*ab(p2, p3))/(ab(p1, p4)**2*ab(p2, p3)**2 + 2*ab(p1, p4)*ab(p2, p1)*ab(p2, p3)*ab(p4, p3) + ab(p2, p1)**2*ab(p4, p3)**2)'
    # input_eq = '-(ab(p1,p2)**2*(ab(p1,p2)*sb(p1,p2) + ab(p2,p3)*sb(p2,p3))*sb(p3,p4))/(ab(p1,p3)*ab(p1,p4)*ab(p2,p3)*sb(p1,p2)*sb(p1,p3))'
    params = AttrDict({

        # Name
        'exp_name': 'Test_eval_spin_hel',
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/scratch/',
        'exp_id': 'test',
        'save_periodic': 0,
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'max_npt': 8,
        'max_scale': 0.5,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': False,
        'int_base': 10,
        'max_len': 512,
        'canonical_form': True,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/checkpoint.pth',
        # 'reload_model': '',

        # Trainer param
        'export_data': False,
        'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.test',
        # 'reload_data': '',
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
        'beam_size': 5,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '/Users/aurelien/Documents/Package_lib/Spinors-1.0',

    })

    check_model_params(params)

    # Start the logger
    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()

    environment.utils.CUDA = not params.cpu

    # Load the model and environment
    env = build_env(params)

    modules = build_modules(env, params)

    # start the wolfram session
    if params.numerical_check:
        session = initialize_numerical_check(params.max_npt, lib_path=params.lib_path)
        env.session = session

    first_num = test_model_expression(env, modules, input_eq, verbose=True, latex_form=True)
    print(first_num)

    if params.numerical_check:
        env.session.stop()
