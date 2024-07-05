"""
Helper functions for the contrastive and simplifier transformers simplification
"""
import torch
import numpy as np
import sympy as sp
from environment.utils import to_cuda, convert_sp_forms, revert_sp_form, reorder_expr
from add_ons.numerical_evaluations import check_numerical_equiv_local
from model.contrastive_learner import build_modules_contrastive
from model import build_modules


def load_modules(envir_c, envir_s, params_c, params_s):
    """
    Given the simplifier and contrastive environments and parameters we construct and load the appropriate
    transformer models
    :param envir_c:
    :param envir_s:
    :param params_c:
    :param params_s:
    :return:
    """

    module_contrastive = build_modules_contrastive(envir_c, params_c)
    encoder_c = module_contrastive['encoder_c']
    encoder_c.eval()

    modules_simplifier = build_modules(envir_s, params_s)
    encoder_s = modules_simplifier['encoder']
    decoder_s = modules_simplifier['decoder']
    encoder_s.eval()
    decoder_s.eval()

    return encoder_c, encoder_s, decoder_s


def load_equation(input_equation, envir, params):
    """
    Given an initial input equation given in str form we isolate the numerator terms
    :param envir:
    :param input_equation:
    :param params:
    :return:
    """
    f = sp.parse_expr(input_equation, local_dict=envir.func_dict)
    if params.canonical_form:
        f = reorder_expr(f)
    return f


def one_hot_encode_sp(sp_equation, envir):
    """
    Take a sympy equation and an associated environment and return a one-hot encoded
    version of the equation that can be passed through a transformer
    :param sp_equation:
    :param envir:
    :return:
    """
    f_prefix = envir.sympy_to_prefix(sp_equation)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir.eos_index] + [envir.word2id[w] for w in x1_prefix] + [envir.eos_index]).view(-1, 1)
    len1 = torch.LongTensor([len(x1)])

    return x1, len1


def one_shot_simplify(envir, module_transfo, f_eq, params_in, blind_const=False, rng=None):
    """
    Test the capacity of the transformer model to resolve a given input
    :param envir:
    :param module_transfo:
    :param f_eq:
    :param params_in:
    :param blind_const
    :param rng:
    :return:
    """

    # Load the transformer models
    encoder, decoder = module_transfo

    # Convert the sympy equation to prefix and one hot encode it - save its length and push it to device
    try:
        # If we blind constants we normalize each of the numerator terms
        if blind_const:
            eq_to_simplify, const_list = blind_constants(f_eq)
        else:
            eq_to_simplify = f_eq
            const_list = None
        x1, len1 = one_hot_encode_sp(eq_to_simplify, envir)
    except AssertionError:
        raise AssertionError("Equation not encoded correctly with amplitude type selected !")
    x1, len1 = to_cuda(x1, len1)

    # Recover the sympy version of the input equation apt for the numerical check
    f_sp = revert_sp_form(f_eq)

    # forward pass of the encoder network
    encoded = encoder('fwd', x=x1, lengths=len1, causal=False)

    # Beam decoding or nucleus sampling
    beam_sz, sample_method, nucleus_prob, temp = params_in

    with torch.no_grad():
        # Generate a greedy solution
        if sample_method == 'Greedy Decoding':
            greedy_sol, _ = decoder.generate(encoded.transpose(0, 1), src_len=len1, max_len=2048,
                                             sample_temperature=None)
            hypotheses = [(0.0, greedy_sol[:-1].squeeze())]
        else:
            # Generate multiple candidate solutions
            nucleus_sample = sample_method == 'Nucleus Sampling'
            _, _, beam = decoder.generate_beam(encoded.transpose(0, 1), len1, beam_size=beam_sz,
                                               length_penalty=1,
                                               early_stopping=True,
                                               max_len=2048,
                                               stochastic=nucleus_sample,
                                               nucl_p=nucleus_prob,
                                               temperature=temp, rng_gen=rng)
            assert len(beam) == 1
            hypotheses = beam[0].hyp
            assert len(hypotheses) == beam_sz

    out_hyp = []

    # Iterate through the generated hypothesis, parse them and check their accuracy
    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):

        # parse decoded hypothesis
        ids = sent[1:].tolist()  # decoded token IDs
        tok = [envir.id2word[wid] for wid in ids]  # convert to prefix

        try:

            # convert to sympy expressions before checking if they are numerically equivalent
            hyp = envir.prefix_to_infix(tok)
            hyp = envir.infix_to_sympy(hyp)
            hyp_disp = convert_sp_forms(hyp, envir.func_dict)

            if blind_const:
                min_const = min(abs(const_list))
                hyp_disp = sp.cancel(f_eq - eq_to_simplify * min_const + hyp_disp * min_const)
                hyp = revert_sp_form(hyp_disp)

            npt = envir.npt_list[0] if len(envir.npt_list) == 1 else None
            matches, rel_diff = check_numerical_equiv_local(envir.special_tokens, hyp, f_sp,  npt=npt)
            out_hyp.append((matches, hyp_disp, rel_diff))
        except:
            pass

    return list(dict.fromkeys(out_hyp))


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


def count_numerator_terms(expression):
    """
    Given an input expression we look at the number of terms in the numerator
    :param expression:
    :return:
    """

    numerator, denominator = sp.fraction(expression)

    if isinstance(numerator, sp.Add):
        num_terms = len(numerator.args)
    elif numerator == 0:
        num_terms = 0
    elif len(numerator.atoms(sp.Add)) == 0:
        num_terms = 1
    else:
        raise ValueError('Could not determine the number of terms in the numerator')
    return num_terms


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


def all_one_shot_simplify(inference_methods, envir, module_transfo, f_eq, params_in, blind_const=False, rng=None):
    """
    Iterate through each inference method and retain the generated hypothesis
    :param inference_methods:
    :param envir:
    :param module_transfo:
    :param f_eq:
    :param params_in:
    :param blind_const:
    :param rng:
    :return:
    """
    hyps_found = []
    for inference_method in inference_methods:
        beam_size, nucleus_p, temperature = params_in
        params_input = (beam_size, inference_method, nucleus_p, temperature)
        hyp_found = one_shot_simplify(envir, module_transfo, f_eq, params_input, blind_const=blind_const, rng=rng)
        hyps_found.extend(hyp_found)

    return list(dict.fromkeys(hyps_found))


def retain_valid_hypothesis(hyps_list, term_init, rng_active):
    """
    Go through a list of generated hypotheses and retain the shortest one (or the one with the smallest constants)
    :param hyps_list:
    :param term_init:
    :return:
    """
    solution_returned = None
    _, const_list_init = blind_constants(term_init)
    min_terms = len(const_list_init)
    min_const_mag = abs(const_list_init).sum()

    for (match, hyp, diff) in hyps_list:

        # If the relative difference is 4 we just have the wrong overall sign (since we check on two datasets)
        if diff == 4:
            hyp = -hyp
            match = True
        if match:
            _, const_list_new = blind_constants(hyp)
            num_terms_new = len(const_list_new)
            new_const_mag = abs(const_list_new).sum()
            if num_terms_new <= min_terms and ((new_const_mag <= min_const_mag and rng_active) or
                                               new_const_mag < min_const_mag):
                min_terms = num_terms_new
                min_const_mag = new_const_mag
                solution_returned = hyp
        else:
            pass

    return solution_returned
