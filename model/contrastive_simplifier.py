"""
Methods relevant for simplifying a large spinor helicity amplitude using both the contrastive model and the simplier
"""

import numpy as np
import sympy as sp
from model.contrastive_learner import build_modules_contrastive
from environment.utils import convert_sp_forms
from model import build_modules
from environment.utils import reorder_expr, to_cuda
from add_ons.mathematica_utils import sp_to_mma, check_numerical_equiv
import torch
import torch.nn as nn


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


def load_equation(envir, input_equation, params):
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
    f = f.cancel()
    numerator, denominator = sp.fraction(f)

    if isinstance(numerator, sp.Add):
        terms = np.asarray(numerator.args)
    elif isinstance(numerator, sp.Mul):
        terms = np.asarray(numerator.args[1].args)
    else:
        terms = [numerator]
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
    elif len(numerator.atoms(sp.Add)) == 0:
        num_terms = 1
    else:
        raise ValueError('Could not determine the number of terms in the numerator')
    return num_terms


def encode_term(envir_c, term, encoder_c):
    """
    Given an input sympy term we convert it to prefix notation and pass it through the
    encoder network of the contrastive learning setup
    :param envir_c:
    :param term:
    :param encoder_c:
    :return:
    """

    t_prefix = envir_c.sympy_to_prefix(term)
    t_in = to_cuda(torch.LongTensor([envir_c.eos_index] + [envir_c.word2id[w] for w in t_prefix] +
                                    [envir_c.eos_index]).view(-1, 1))[0]
    len_in = to_cuda(torch.LongTensor([len(t_in)]))[0]

    # Forward
    with torch.no_grad():
        encoded = encoder_c('fwd', x=t_in, lengths=len_in, causal=False)

    return encoded


def encode_s_term(envir_s, term, encoder_s):
    """
    Given an input sympy term we convert it to prefix notation and pass it through the
    encoder network of the simplifier encoder part of the transformer model
    :param envir_s:
    :param term:
    :param encoder_s:
    :return:
    """

    f_prefix = envir_s.sympy_to_prefix(term)
    x1_prefix = f_prefix
    x1 = torch.LongTensor([envir_s.eos_index] + [envir_s.word2id[w] for w in x1_prefix] + [envir_s.eos_index]).view(-1,
                                                                                                                    1)
    len1 = torch.LongTensor([len(x1)])
    x1, len1 = to_cuda(x1, len1)

    # Forward
    with torch.no_grad():
        encoded_s = encoder_s('fwd', x=x1, lengths=len1, causal=False)

    return encoded_s, len1


def masked_similarity_term(envir_c, term_ref, terms_comp, encoder_c):
    """
    Given a reference term we compute its cosine similarity with a list of target terms.
    We calculate the cosine similarity only on the parts of the terms that do not
    share similar factors in common
    :param envir_c:
    :param term_ref:
    :param terms_comp:
    :param encoder_c:
    :return:
    """
    metric_sim = nn.CosineSimilarity(dim=-1)
    similarity_vect = torch.zeros(len(terms_comp))
    for i, term in enumerate(terms_comp):

        newterm_ref, newterm_comp = sp.fraction(sp.cancel(term_ref / term))

        encoded_ref = encode_term(envir_c, newterm_ref, encoder_c)
        encoded_comp = encode_term(envir_c, newterm_comp, encoder_c)

        similarity_vect[i] = metric_sim(encoded_ref, encoded_comp)

    return similarity_vect


def similarity_terms(envir_c, terms, encoder_c):
    """
    Given a list of terms we compute the associated cosine similarity matrix
    No masking is assumed here and we compare the complete form of the terms
    :param envir_c:
    :param terms:
    :param encoder_c:
    :return:
    """
    metric_sim = nn.CosineSimilarity(dim=-1)
    encoded_terms = [encode_term(envir_c, term, encoder_c) for term in terms]

    similarity_mat = metric_sim(torch.stack(encoded_terms), torch.transpose(torch.stack(encoded_terms), 0, 1))

    return similarity_mat


def find_single_simplification_terms(similarity_vect, terms_comp, cutoff=0.95, denominator=None):
    """
    Given a similarity vector we retain the expression that holds the greatest promise in simplifying
    :param similarity_vect:
    :param terms_comp:
    :param cutoff:
    :param denominator:
    :return:
    """
    mask = similarity_vect > cutoff

    # If no terms are close enough then we assume that we cannot simplify
    # Higher than 1 as we have similarity one with the reference
    if not torch.sum(mask).item() > 1:
        return None

    # Retain only the terms that have a high chance of simplifying and combine them
    relevant_terms = terms_comp[mask]
    terms_to_simplify = relevant_terms.sum()

    if denominator is not None:
        terms_to_simplify = terms_to_simplify / denominator

    return terms_to_simplify


def convert_generated_sol(envir_s, sol_toks):
    """
    Convert the generated tokens into sympy form
    :param envir_s:
    :param sol_toks:
    :return:
    """
    # parse decoded hypothesis
    ids = sol_toks[1:].tolist()  # decoded token IDs
    tok = [envir_s.id2word[wid] for wid in ids]  # convert to prefix

    # Parse the token and convert to the relevant format
    hyp = envir_s.prefix_to_infix(tok)
    hyp_sp = envir_s.infix_to_sympy(hyp)  # convert to SymPy

    return hyp_sp


def greedy_decoding(encoded, decoder_s, len_in, params_s, envir_s):
    """
    Generate a single greedy solution
    :param encoded:
    :param decoder_s:
    :param len_in:
    :param params_s:
    :param envir_s
    :return:
    """
    greedy_sol, _ = decoder_s.generate(encoded.transpose(0, 1), src_len=len_in,
                                       max_len=params_s.max_len, sample_temperature=None)

    sp_greedy_sol = convert_generated_sol(envir_s, greedy_sol)

    return sp_greedy_sol


def check_valid_solution(terms_to_simplify, num_terms_init, hyp_sp, envir_s, params_s):
    """
    Check if our proposed solution is correct (also look if it is valid up to an overall sign)
    :param terms_to_simplify:
    :param num_terms_init:
    :param hyp_sp:
    :param envir_s:
    :param params_s:
    :return:
    """
    # Check if we are indeed decreasing the length of the expression
    num_terms_hyp = count_numerator_terms(hyp_sp)
    if num_terms_hyp >= num_terms_init:
        return False, None, None

    # Do the numerical equivalence check
    hyp_mma = sp_to_mma(hyp_sp, envir_s.npt_list, params_s.bracket_tokens, envir_s.func_dict)
    f_sp = envir_s.infix_to_sympy(envir_s.prefix_to_infix(envir_s.sympy_to_prefix(terms_to_simplify)))
    tgt_mma = sp_to_mma(f_sp, envir_s.npt_list, params_s.bracket_tokens, envir_s.func_dict)
    matches, error = check_numerical_equiv(envir_s.session, hyp_mma, tgt_mma)

    # Hypothesis agrees numerically
    if matches:
        return True, hyp_sp, num_terms_hyp
    # Hypothesis is actually correct up to an overall sign
    elif error == -1:
        return True, -hyp_sp, num_terms_hyp
    # Hypothesis does not match
    else:
        return False, None, None


def generate_beam_hyp(encoded, len_in, decoder_s, params_s):
    """
    Generate the beam of hypothesis to be tested
    :param encoded:
    :param len_in:
    :param decoder_s:
    :param params_s:
    :return:
    """

    # Beam decoding - Start with deterministic setup
    with torch.no_grad():
        _, _, beam = decoder_s.generate_beam(encoded.transpose(0, 1), len_in, beam_size=params_s.beam_size,
                                             length_penalty=params_s.beam_length_penalty,
                                             early_stopping=params_s.beam_early_stopping,
                                             max_len=params_s.max_len,
                                             stochastic=False,
                                             nucl_p=params_s.nucleus_p,
                                             temperature=params_s.temperature)
    hypotheses = beam[0].hyp

    return hypotheses


def generate_nucleus_hyp(encoded, len_in, decoder_s, params_s):
    """
    Use nucleus sampling to generate the list of hypothesis to be tested
    :param encoded:
    :param len_in:
    :param decoder_s:
    :param params_s:
    :return:
    """
    # Nucleus decoding - Use a stochastic setup
    with torch.no_grad():
        _, _, beam = decoder_s.generate_beam(encoded.transpose(0, 1), len_in, beam_size=params_s.beam_size,
                                             length_penalty=params_s.beam_length_penalty,
                                             early_stopping=params_s.beam_early_stopping,
                                             max_len=params_s.max_len,
                                             stochastic=True,
                                             nucl_p=params_s.nucleus_p,
                                             temperature=params_s.temperature)
    hypotheses = beam[0].hyp

    return hypotheses


def check_hypothesis(terms_to_simplify, num_terms_init, hypotheses, envir_s, params_s):
    """
    For a set of hypothesis we check if any single one is correct
    If multiple are correct then we return the one that simplifies the equation the most
    :param terms_to_simplify:
    :param num_terms_init:
    :param hypotheses:
    :param envir_s:
    :param params_s:
    :return:
    """
    nterms_min = num_terms_init
    solution = None
    valid = False

    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):
        hyp = convert_generated_sol(envir_s, sent[1:])
        val, sol, nterms = check_valid_solution(terms_to_simplify, num_terms_init, hyp, envir_s, params_s)

        if val and nterms < nterms_min:
            solution = sol
            nterms_min = nterms
            valid = True

    return valid, solution


def attempt_simplification(terms_to_simplify, encoder_s, decoder_s, envir_s, params_s):
    """
    Given a term to simplify we try and use our simplifier module to simplify it.
    We try in order
    1) Greedy decoding
    2) Beam search
    3) Nucleus sampling

    If any of these methods succeed we return the solution found

    :param terms_to_simplify:
    :param encoder_s:
    :param decoder_s:
    :param envir_s:
    :param params_s:
    :return:
    """
    # Stats for the initial term
    terms_to_simplify = terms_to_simplify.cancel()
    num_terms_init = count_numerator_terms(terms_to_simplify)

    # Encode the term to simplify
    encoded_term, len1 = encode_s_term(envir_s, terms_to_simplify, encoder_s)

    # Greedy decoding - if we solve with it directly we return the solution
    greedy_sol = greedy_decoding(encoded_term, decoder_s, len1, params_s, envir_s)
    valid, solution, _ = check_valid_solution(terms_to_simplify, num_terms_init, greedy_sol, envir_s, params_s)
    if valid:
        return convert_sp_forms(solution, envir_s.func_dict)

    # Beam decoding
    beam_hypothesis = generate_beam_hyp(encoded_term, len1, decoder_s, params_s)
    valid, solution = check_hypothesis(terms_to_simplify, num_terms_init, beam_hypothesis, envir_s, params_s)
    if valid:
        return convert_sp_forms(solution, envir_s.func_dict)

    # Nucleus sampling
    beam_hypothesis = generate_nucleus_hyp(encoded_term, len1, decoder_s, params_s)
    valid, solution = check_hypothesis(terms_to_simplify, num_terms_init, beam_hypothesis, envir_s, params_s)
    if valid:
        return convert_sp_forms(solution, envir_s.func_dict)
