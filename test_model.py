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

        # Parse the identities if required
        try:
            hyp = envir.prefix_to_infix(tok)
            if '&' in tok:
                prefix_info = tok[tok.index('&'):]
                info_infix = env.scr_prefix_to_infix(prefix_info)
            else:
                info_infix = ''

            # convert to infix
            hyp = envir.infix_to_sympy(hyp)  # convert to SymPy

            # When integrating the symbol
            if params.numerical_check:
                hyp_mma = sp_to_mma(hyp, params.bracket_tokens, env.func_dict)
                f_sp = env.infix_to_sympy(env.prefix_to_infix(env.sympy_to_prefix(f)))
                tgt_mma = sp_to_mma(f_sp, params.bracket_tokens, env.func_dict)
                matches, _ = check_numerical_equiv(envir.session, hyp_mma, tgt_mma)
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
            info_infix = ''

        if verbose:
            # print result
            if latex_form:
                print("%.5f  %s %s %s %s" % (score, res, hyp, info_infix, latex(convert_sp_forms(hyp, envir.func_dict))))
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

    # Example with integrating the symbol (don't need to precise the target, we can check it with MMA)

    # Example 1 : A long 6 pt amplitude
    # input_eq = '(-ab(1,2)**2*ab(1,6)**2*sb(1,4)**2*sb(1,6)**2*sb(3,6)**2 - 2*ab(1,2)**2*ab(1,6)*ab(2,6)*sb(1,4)*sb(1,6)**2*sb(2,6)*sb(3,4)*sb(3,6) - 2*ab(1,2)**2*ab(1,6)*ab(3,6)*sb(1,4)**2*sb(1,6)*sb(3,6)**3 - 2*ab(1,2)**2*ab(1,6)*ab(4,6)*sb(1,4)**2*sb(1,6)*sb(3,6)**2*sb(4,6) - 2*ab(1,2)**2*ab(1,6)*ab(5,6)*sb(1,4)**2*sb(1,6)*sb(3,6)**2*sb(5,6) - ab(1,2)**2*ab(2,6)**2*sb(1,6)**2*sb(2,6)**2*sb(3,4)**2 - 2*ab(1,2)**2*ab(2,6)*ab(3,6)*sb(1,4)*sb(1,6)*sb(2,6)*sb(3,4)*sb(3,6)**2 - 2*ab(1,2)**2*ab(2,6)*ab(4,6)*sb(1,4)*sb(1,6)*sb(2,6)*sb(3,4)*sb(3,6)*sb(4,6) - 2*ab(1,2)**2*ab(2,6)*ab(5,6)*sb(1,4)*sb(1,6)*sb(2,6)*sb(3,4)*sb(3,6)*sb(5,6) - ab(1,2)**2*ab(3,6)**2*sb(1,4)**2*sb(3,6)**4 - 2*ab(1,2)**2*ab(3,6)*ab(4,6)*sb(1,4)**2*sb(3,6)**3*sb(4,6) - 2*ab(1,2)**2*ab(3,6)*ab(5,6)*sb(1,4)**2*sb(3,6)**3*sb(5,6) - ab(1,2)**2*ab(4,6)**2*sb(1,4)**2*sb(3,6)**2*sb(4,6)**2 - 2*ab(1,2)**2*ab(4,6)*ab(5,6)*sb(1,4)**2*sb(3,6)**2*sb(4,6)*sb(5,6) - ab(1,2)**2*ab(5,6)**2*sb(1,4)**2*sb(3,6)**2*sb(5,6)**2)/(ab(1,6)**2*ab(2,4)**2*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)**3 + 2*ab(1,6)**2*ab(2,4)*ab(2,5)*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)**2*sb(3,5) + 2*ab(1,6)**2*ab(2,4)*ab(2,6)*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)**2*sb(3,6) + ab(1,6)**2*ab(2,5)**2*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)*sb(3,5)**2 + 2*ab(1,6)**2*ab(2,5)*ab(2,6)*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)*sb(3,5)*sb(3,6) + ab(1,6)**2*ab(2,6)**2*ab(5,6)*sb(1,2)*sb(1,6)**2*sb(3,4)*sb(3,6)**2 + 2*ab(1,6)*ab(2,4)**2*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**3*sb(3,6) + 2*ab(1,6)*ab(2,4)**2*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**3*sb(4,6) + 2*ab(1,6)*ab(2,4)**2*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)**3*sb(5,6) + 4*ab(1,6)*ab(2,4)*ab(2,5)*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,5)*sb(3,6) + 4*ab(1,6)*ab(2,4)*ab(2,5)*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,5)*sb(4,6) + 4*ab(1,6)*ab(2,4)*ab(2,5)*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,5)*sb(5,6) + 4*ab(1,6)*ab(2,4)*ab(2,6)*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,6)**2 + 4*ab(1,6)*ab(2,4)*ab(2,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,6)*sb(4,6) + 4*ab(1,6)*ab(2,4)*ab(2,6)*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)**2*sb(3,6)*sb(5,6) + 2*ab(1,6)*ab(2,5)**2*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)**2*sb(3,6) + 2*ab(1,6)*ab(2,5)**2*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)**2*sb(4,6) + 2*ab(1,6)*ab(2,5)**2*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)**2*sb(5,6) + 4*ab(1,6)*ab(2,5)*ab(2,6)*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)*sb(3,6)**2 + 4*ab(1,6)*ab(2,5)*ab(2,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)*sb(3,6)*sb(4,6) + 4*ab(1,6)*ab(2,5)*ab(2,6)*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,5)*sb(3,6)*sb(5,6) + 2*ab(1,6)*ab(2,6)**2*ab(3,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,6)**3 + 2*ab(1,6)*ab(2,6)**2*ab(4,6)*ab(5,6)*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,6)**2*sb(4,6) + 2*ab(1,6)*ab(2,6)**2*ab(5,6)**2*sb(1,2)*sb(1,6)*sb(3,4)*sb(3,6)**2*sb(5,6) + ab(2,4)**2*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**3*sb(3,6)**2 + 2*ab(2,4)**2*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)**3*sb(3,6)*sb(4,6) + 2*ab(2,4)**2*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**3*sb(3,6)*sb(5,6) + ab(2,4)**2*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**3*sb(4,6)**2 + 2*ab(2,4)**2*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**3*sb(4,6)*sb(5,6) + ab(2,4)**2*ab(5,6)**3*sb(1,2)*sb(3,4)**3*sb(5,6)**2 + 2*ab(2,4)*ab(2,5)*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(3,6)**2 + 4*ab(2,4)*ab(2,5)*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(3,6)*sb(4,6) + 4*ab(2,4)*ab(2,5)*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(3,6)*sb(5,6) + 2*ab(2,4)*ab(2,5)*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(4,6)**2 + 4*ab(2,4)*ab(2,5)*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(4,6)*sb(5,6) + 2*ab(2,4)*ab(2,5)*ab(5,6)**3*sb(1,2)*sb(3,4)**2*sb(3,5)*sb(5,6)**2 + 2*ab(2,4)*ab(2,6)*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,6)**3 + 4*ab(2,4)*ab(2,6)*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,6)**2*sb(4,6) + 4*ab(2,4)*ab(2,6)*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**2*sb(3,6)**2*sb(5,6) + 2*ab(2,4)*ab(2,6)*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)**2*sb(3,6)*sb(4,6)**2 + 4*ab(2,4)*ab(2,6)*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)**2*sb(3,6)*sb(4,6)*sb(5,6) + 2*ab(2,4)*ab(2,6)*ab(5,6)**3*sb(1,2)*sb(3,4)**2*sb(3,6)*sb(5,6)**2 + ab(2,5)**2*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(3,6)**2 + 2*ab(2,5)**2*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(3,6)*sb(4,6) + 2*ab(2,5)**2*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(3,6)*sb(5,6) + ab(2,5)**2*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(4,6)**2 + 2*ab(2,5)**2*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(4,6)*sb(5,6) + ab(2,5)**2*ab(5,6)**3*sb(1,2)*sb(3,4)*sb(3,5)**2*sb(5,6)**2 + 2*ab(2,5)*ab(2,6)*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)**3 + 4*ab(2,5)*ab(2,6)*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)**2*sb(4,6) + 4*ab(2,5)*ab(2,6)*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)**2*sb(5,6) + 2*ab(2,5)*ab(2,6)*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)*sb(4,6)**2 + 4*ab(2,5)*ab(2,6)*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)*sb(4,6)*sb(5,6) + 2*ab(2,5)*ab(2,6)*ab(5,6)**3*sb(1,2)*sb(3,4)*sb(3,5)*sb(3,6)*sb(5,6)**2 + ab(2,6)**2*ab(3,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,6)**4 + 2*ab(2,6)**2*ab(3,6)*ab(4,6)*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,6)**3*sb(4,6) + 2*ab(2,6)**2*ab(3,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,6)**3*sb(5,6) + ab(2,6)**2*ab(4,6)**2*ab(5,6)*sb(1,2)*sb(3,4)*sb(3,6)**2*sb(4,6)**2 + 2*ab(2,6)**2*ab(4,6)*ab(5,6)**2*sb(1,2)*sb(3,4)*sb(3,6)**2*sb(4,6)*sb(5,6) + ab(2,6)**2*ab(5,6)**3*sb(1,2)*sb(3,4)*sb(3,6)**2*sb(5,6)**2)'

    # Example 2: A 4pt ampltiude that should reduce to the Parke Taylor formula
    # input_eq = "((-(ab(1,4)*ab(2,1)*ab(2,3)*sb(1,2)*sb(2,3)) + ab(1,2)*ab(2,3)*ab(4,1)*sb(1,2)*sb(2,3) + ab(2,3)**2*ab(4,1)*sb(2,3)**2 - ab(1,3)*ab(2,1)*ab(2,4)*sb(1,2)*sb(3,2) - ab(1,3)*ab(2,3)*ab(4,1)*sb(1,3)*sb(3,2))*sb(3,4))/(ab(2,3)*ab(3,4)*ab(4,1)*sb(1,2)**2*sb(2,3))"

    # Example 3: Almost simplified versions of the Parke Taylor formula at 4 pt
    # input_eq = '(ab(1,2)*sb(3,4)**2)/(ab(1,4)*sb(1,2)*sb(1,4))'
    # input_eq = '(ab(1,2)**2*sb(3,4))/(ab(1,4)*sb(1,2)*ab(2,3))'
    # input_eq = '(ab(1,2)*sb(3,4)**2)/(ab(2,3)*sb(1,2)*sb(2,3))'
    # input_eq = '(ab(1,2)**2*(-(ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4)) - ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) + ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,5)*sb(3,4)**2 - ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,3)*ab(2,3)*sb(1,2)**2*sb(2,3)**2*sb(3,5) + ab(1,2)*ab(1,3)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)**2*sb(3,5) + 2*ab(1,2)*ab(1,3)*ab(2,3)**2*sb(1,2)*sb(2,3)**3*sb(3,5) + ab(1,3)**2*ab(2,3)**2*sb(1,3)*sb(2,3)**3*sb(3,5) + ab(1,3)*ab(2,3)**3*sb(2,3)**4*sb(3,5) + ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(2,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) + ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)**2*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5)))/(ab(1,3)*ab(1,4)*ab(1,5)*ab(2,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*(ab(1,2)*sb(1,2) + ab(1,3)*sb(1,3) + ab(2,3)*sb(2,3))*(ab(2,3)*sb(2,3) + ab(2,4)*sb(2,4) + ab(3,4)*sb(3,4)))'
    # input_eq = '((-(ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4)) - ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4) - 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,5)*sb(3,4) - ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) - ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(2,5)*sb(3,4) + ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,5)*sb(3,4)**2 + ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,5)*sb(3,4)**2 + ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,5)*sb(3,4)**2 - ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,5)*sb(3,4)**2 - ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,5)*sb(3,4)**2 + ab(1,2)**2*ab(1,3)*ab(2,3)*sb(1,2)**2*sb(2,3)**2*sb(3,5) + ab(1,2)*ab(1,3)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)**2*sb(3,5) + 2*ab(1,2)*ab(1,3)*ab(2,3)**2*sb(1,2)*sb(2,3)**3*sb(3,5) + ab(1,3)**2*ab(2,3)**2*sb(1,3)*sb(2,3)**3*sb(3,5) + ab(1,3)*ab(2,3)**3*sb(2,3)**4*sb(3,5) + ab(1,2)**2*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**3*ab(3,4)*sb(1,2)**2*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + ab(1,2)**2*ab(1,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,5) + 2*ab(1,2)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + 2*ab(1,2)**2*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)**2*sb(2,4)*sb(3,5) + ab(1,4)*ab(2,3)**3*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(2,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(2,4)*sb(3,5) + ab(1,2)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,2)**2*ab(2,4)*ab(3,4)*sb(1,2)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(2,4)*sb(1,3)*sb(2,3)*sb(2,4)**2*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(2,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) + ab(1,2)*ab(2,3)*ab(2,4)*ab(3,4)*sb(2,3)**2*sb(2,4)**2*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(2,3)*sb(1,2)**2*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)**2*ab(1,4)*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,3)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)*ab(2,3)**2*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,2)*ab(1,3)*ab(2,3)*ab(3,4)*sb(1,2)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)**2*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)**2*sb(3,4)*sb(3,5) + ab(1,3)*ab(2,3)**2*ab(3,4)*sb(2,3)**3*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,4)**2*ab(2,3)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)**2*ab(1,4)*ab(3,4)*sb(1,2)**2*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,3)*ab(1,4)**2*ab(2,3)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,2)*ab(1,3)*ab(1,4)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,4)*sb(3,4)*sb(3,5) - ab(1,4)**2*ab(2,3)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)**2*ab(3,4)**2*sb(1,2)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,3)*ab(1,4)*ab(2,3)*ab(3,4)*sb(1,3)*sb(2,3)*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,4)*ab(2,3)**2*ab(3,4)*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5) + ab(1,2)*ab(2,3)*ab(3,4)**2*sb(2,3)**2*sb(2,4)*sb(3,4)*sb(3,5)))/(ab(1,3)*ab(1,4)*ab(1,5)*ab(2,3)*ab(3,4)*sb(1,2)*sb(1,3)*sb(2,3)*(ab(1,2)*sb(1,2) + ab(1,3)*sb(1,3) + ab(2,3)*sb(2,3))*(ab(2,3)*sb(2,3) + ab(2,4)*sb(2,4) + ab(3,4)*sb(3,4)))'
    input_eq = '(-ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,4)**2*sb(2,3)*sb(2,5)**2 + 2*ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)*sb(2,4)*sb(2,5) - ab(1,2)*ab(2,3)*ab(2,4)*ab(3,5)*ab(4,5)*sb(1,5)**2*sb(2,3)*sb(2,4)**2)/sb(4,5)**2'

    parameters = AttrDict({
        # Experiment Name
        'exp_name': 'Test_eval_spin_hel',
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/scratch/',
        'exp_id': 'test',
        'save_periodic': 0,
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'max_npt': 6,
        'max_scale': 2,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': False,
        'int_base': 10,
        'max_len': 2048,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
        'l_scale': 0.75,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt6/checkpoint.pth',
        # 'reload_model': '',

        # Trainer param
        'export_data': False,
        # 'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8/data.prefix.counts.test',
        'reload_data': '',
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
        'beam_size': 10,
        'beam_length_penalty': 1,
        'beam_early_stopping': False,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '/Users/aurelien/Documents/Package_lib/Spinors-1.0',

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
        session = initialize_numerical_check(parameters.max_npt, lib_path=parameters.lib_path)
        env.session = session

    first_num = test_model_expression(env, modules, input_eq, parameters, verbose=True, latex_form=True)
    print(first_num)

    if parameters.numerical_check:
        env.session.stop()

