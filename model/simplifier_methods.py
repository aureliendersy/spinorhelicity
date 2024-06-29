"""
Helper functions for the contrastive and simplifier transformers
"""
import torch
import numpy as np
import sympy as sp
from environment.utils import to_cuda, convert_sp_forms
from add_ons.numerical_evaluations import check_numerical_equiv_local


def test_model_expression(envir, module_transfo, f_eq, params_in):
    """
    Test the capacity of the transformer model to resolve a given input
    :param envir:
    :param module_transfo:
    :param input_equation:
    :param params:
    :return:
    """

    encoder, decoder = module_transfo
    f_prefix = envir.sympy_to_prefix(f_eq)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in x1_prefix] + [envir.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # forward
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding
    beam_sz, nucleus_sample, nucleus_prob, temp = params_in
    with torch.no_grad():
        _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_sz,
                                           length_penalty=1,
                                           early_stopping=True,
                                           max_len=2048,
                                           stochastic=nucleus_sample,
                                           nucl_p=nucleus_prob,
                                           temperature=temp)
        assert len(beam) == 1
    hypotheses = beam[0].hyp
    assert len(hypotheses) == beam_sz

    out_hyp = []
    # Print out the scores and the hypotheses
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [envir.id2word[wid] for wid in ids]  # convert to prefix

        # Parse the identities if required
        try:
            hyp = envir.prefix_to_infix(tok)

            # convert to infix
            hyp = envir.infix_to_sympy(hyp)  # convert to SymPy
            hyp_disp = convert_sp_forms(hyp, envir.func_dict)
            f_sp = envir.infix_to_sympy(envir.prefix_to_infix(envir.sympy_to_prefix(f_eq)))
            npt = envir.npt_list[0] if len(envir.npt_list) == 1 else None
            matches, _ = check_numerical_equiv_local(envir.special_tokens, hyp, f_sp,  npt=npt)
            out_hyp.append((matches, hyp_disp))
        except:
            pass

    return list(set(out_hyp))


def extract_num_denom(input_eq):
    """
    Given a sympy equation we recover the numerator and the denominator
    :param input_eq:
    :return:
    """
    f = input_eq.cancel()
    numerator, denominator = sp.fraction(f)

    # Return the numerator as a list of terms
    if isinstance(numerator, sp.Add):
        terms = np.asarray(numerator.args)
    else:
        terms = np.asarray([numerator])
    return terms, denominator


def blind_constants(input_expression):
    """
    Given an input equation we isolate the numerator terms and return the expression
    with all constants set to 1 or -1 depending on the sign. Also return the list of constants
    :param input_expression:
    :return:
    """
    num, denom = extract_num_denom(input_expression)
    new_num = []
    const_list = []
    for term in num:
        if isinstance(term, sp.Add) or isinstance(term, sp.Mul):
            # constant is the overall term in front
            const = [term_mult for term_mult in term.args if isinstance(term_mult, sp.Integer)]
        else:
            const = []
        if len(const) == 0:
            new_num.append(term)
            const_list.append(1)

        # If we find the constant we normalize the term and save it
        elif len(const) == 1:
            new_num.append(term/abs(const[0]))
            const_list.append(const[0])
        else:
            print(num)
            print(const)
            raise ValueError('Found two constants in a numerator term')

    # Return the expression with contants set to +- 1 and the list of original constants
    return (np.array(new_num).sum()) / denom, np.array(const_list)
