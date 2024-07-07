"""
Methods relevant for simplifying a large spinor helicity amplitude using the contrastive model
"""

from logging import getLogger
import numpy as np
import pandas as pd
import sympy as sp
from environment.utils import convert_sp_forms
from environment.utils import to_cuda
from model.simplifier_methods import (blind_constants, extract_num_denom, all_one_shot_simplify,
                                      retain_valid_hypothesis, count_numerator_terms)
from add_ons.mathematica_utils import sp_to_mma, check_numerical_equiv_mma
import torch
import torch.nn as nn

logger = getLogger()
streamlit_logger = getLogger(__name__)
streamlit_logger2 = getLogger(__name__+'2')


def check_for_overall_const(input_eq):
    """
    Check if any of the terms in the input equation is multiplied by an overall constant
    :param input_eq:
    :return:
    """

    numerator, _ = sp.fraction(input_eq)

    if isinstance(numerator, sp.Add):
        terms = np.asarray(numerator.args)
    elif isinstance(numerator, sp.Mul):
        terms = np.asarray(numerator.args[1].args)
    else:
        terms = [numerator]

    return any([any([isinstance(termst, sp.Integer) and abs(termst) > 1 for termst in term.args]) for term in terms])


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


def normalize_term(term_in):
    """
    Given a term in the numerator return the same term normalize (constant 1 or -1)
    :param term_in:
    :return:
    """

    if isinstance(term_in, sp.Integer):
        return term_in/abs(term_in)

    if not isinstance(term_in, sp.Mul):
        raise ValueError('Expected a multiplicative term to normalize')

    const_vect = [term_mult for term_mult in term_in.args if isinstance(term_mult, sp.Integer)]

    if len(const_vect) > 1:
        raise ValueError('Found two constants in a numerator term when normalizing')
    const = 1 if len(const_vect) == 0 else abs(const_vect[0])

    return term_in / const


def masked_similarity_term(envir_c, term_ref, terms_comp, encoder_c, const_blind=False, masked=True):
    """
    Given a reference term we compute its cosine similarity with a list of target terms.
    We calculate the cosine similarity only on the parts of the terms that do not
    share similar factors in common
    :param envir_c:
    :param term_ref:
    :param terms_comp:
    :param encoder_c:
    :param const_blind:
    :param masked:
    :return:
    """
    metric_sim = nn.CosineSimilarity(dim=-1)
    similarity_vect = torch.zeros(len(terms_comp))
    for i, term in enumerate(terms_comp):

        if masked:
            newterm_ref, newterm_comp = sp.fraction(sp.cancel(term_ref / term))
        else:
            newterm_ref, newterm_comp = term_ref, term

        if const_blind:
            newterm_ref = normalize_term(newterm_ref)
            newterm_comp = normalize_term(newterm_comp)

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


def find_single_simplification_terms(similarity_vect, terms_comp, cutoff=0.9, denominator=None, short_search=False):
    """
    Given a similarity vector we retain the expression that holds the greatest promise in simplifying
    :param similarity_vect:
    :param terms_comp:
    :param cutoff:
    :param denominator:
    :param short_search:
    :return:
    """
    mask = similarity_vect > cutoff

    # If no terms are close enough then we assume that we cannot simplify
    # Higher than 1 as we have similarity one with the reference
    if not torch.sum(mask).item() > 1:
        return None, None

    # If we do a short search we limit up to 4 terms at most
    if short_search:
        _, indices_k = torch.topk(similarity_vect, k=min(4, len(similarity_vect)))
        mask2 = torch.zeros_like(mask, dtype=torch.bool)
        mask2[indices_k] = True
        mask = mask & mask2

    # Retain only the terms that have a high chance of simplifying and combine them
    relevant_terms = terms_comp[mask]

    rest_terms = terms_comp[~mask]
    terms_to_simplify = relevant_terms.sum()

    if denominator is not None:
        terms_to_simplify = terms_to_simplify / denominator

    return terms_to_simplify, rest_terms


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

    # Parse the token and convert to the relevant format - If we have an invalid format we return None
    try:
        hyp = envir_s.prefix_to_infix(tok)
        hyp_sp = envir_s.infix_to_sympy(hyp)  # convert to SymPy
    except:
        hyp_sp = None

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

    sp_greedy_sol = convert_generated_sol(envir_s, greedy_sol[:-1, :].squeeze())

    return sp_greedy_sol


def check_valid_solution(terms_to_simplify, num_terms_init, hyp_sp, envir_s, params_s, rng_active_num, ref_term=None):
    """
    Check if our proposed solution is correct (also look if it is valid up to an overall sign)
    :param terms_to_simplify:
    :param num_terms_init:
    :param hyp_sp:
    :param envir_s:
    :param params_s:
    :param rng_active_num:
    :param ref_term:
    :return:
    """

    # For an invalid expression we return False
    if hyp_sp is None:
        return False, None, None

    # Check if we are indeed decreasing the length of the expression
    # If we have an overall constant then we allow expressions with the same length
    num_terms_hyp = count_numerator_terms(hyp_sp)

    if num_terms_hyp is None:
        return False, None, None

    # If we account for constants we now have to compare to the solution once added with the reference term
    if ref_term is not None:

        # Scale the hypothesis by the minimal constant found in original expression
        min_const = min(abs(ref_term[1]))
        hyp_sp_scale = convert_sp_forms(hyp_sp, envir_s.func_dict) * min_const

        # Get the new total hypothesis and corresponding constants
        hyp_adjusted = sp.cancel(ref_term[0] - terms_to_simplify*min_const + hyp_sp_scale)
        _, const_list_new = blind_constants(hyp_adjusted)
        num_terms_hyp = len(const_list_new)

        # Convert to correct format
        hyp_sp = envir_s.infix_to_sympy(envir_s.prefix_to_infix(envir_s.sympy_to_prefix(hyp_adjusted)))
        terms_to_simplify = ref_term[0]

        # If we generate more terms or they are associated with bigger constants we get rid of the expression
        if num_terms_hyp > num_terms_init or (num_terms_hyp == num_terms_init
                                              and abs(const_list_new).sum() >= abs(ref_term[1]).sum()):
            return False, None, None

    elif check_for_overall_const(terms_to_simplify) or rng_active_num:
        if num_terms_hyp > num_terms_init:
            return False, None, None
    else:
        if num_terms_hyp >= num_terms_init:
            return False, None, None

    # Do the numerical equivalence check
    hyp_mma = sp_to_mma(hyp_sp, envir_s.npt_list, params_s.bracket_tokens, envir_s.func_dict)
    f_sp = envir_s.infix_to_sympy(envir_s.prefix_to_infix(envir_s.sympy_to_prefix(terms_to_simplify)))
    tgt_mma = sp_to_mma(f_sp, envir_s.npt_list, params_s.bracket_tokens, envir_s.func_dict)
    matches, error = check_numerical_equiv_mma(envir_s.session, hyp_mma, tgt_mma)

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


def generate_nucleus_hyp(encoded, len_in, decoder_s, params_s, rng):
    """
    Use nucleus sampling to generate the list of hypothesis to be tested
    :param encoded:
    :param len_in:
    :param decoder_s:
    :param params_s:
    :param rng:
    :return:
    """
    rng_active, rng_gens = rng
    # Nucleus decoding - Use a stochastic setup
    with torch.no_grad():
        _, _, beam = decoder_s.generate_beam(encoded.transpose(0, 1), len_in, beam_size=params_s.beam_size,
                                             length_penalty=params_s.beam_length_penalty,
                                             early_stopping=params_s.beam_early_stopping,
                                             max_len=params_s.max_len,
                                             stochastic=True,
                                             nucl_p=params_s.nucleus_p,
                                             temperature=params_s.temperature,
                                             rng_gen=rng_gens[1])
    hypotheses = beam[0].hyp

    return hypotheses


def check_hypothesis(terms_to_simplify, num_terms_init, hypotheses, envir_s, params_s, rng_active_num, ref_term=None):
    """
    For a set of hypothesis we check if any single one is correct
    If multiple are correct then we return the one that simplifies the equation the most
    :param terms_to_simplify:
    :param num_terms_init:
    :param hypotheses:
    :param envir_s:
    :param params_s:
    :param rng_active_num:
    :param ref_term:
    :return:
    """
    nterms_min = num_terms_init
    solution = None
    valid = False

    for num, (score, sent) in enumerate(sorted(hypotheses, key=lambda y: y[0], reverse=True)):
        hyp = convert_generated_sol(envir_s, sent)
        val, sol, nterms = check_valid_solution(terms_to_simplify, num_terms_init, hyp, envir_s, params_s,
                                                rng_active_num, ref_term=ref_term)

        if val and nterms <= nterms_min:
            solution = sol
            nterms_min = nterms
            valid = True

    return valid, solution


def attempt_simplification(terms_to_simplify, encoder_s, decoder_s, envir_s, params_s, rng, const_blind=False):
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
    :param rng:
    :param const_blind:
    :return:
    """

    # Stats for the initial term
    terms_to_simplify = terms_to_simplify.cancel()
    num_terms_init = count_numerator_terms(terms_to_simplify)

    # If we want to be blind to constants
    if const_blind:
        ref_term_init = terms_to_simplify
        logger.info('Reference expression is {}'.format(ref_term_init))
        terms_to_simplify, const_list = blind_constants(terms_to_simplify)
        ref_term = (ref_term_init, const_list)
    else:
        ref_term = None

    logger.info('Attempting to simplify {} terms'.format(num_terms_init))
    logger.info('Expression considered is {}'.format(terms_to_simplify))

    # Encode the term to simplify
    encoded_term, len1 = encode_s_term(envir_s, terms_to_simplify, encoder_s)

    # If rng active we have a chance to retain solution of same length
    rng_active, rng_gens = rng
    rng_active_num = False if not rng_active else rng_gens[0].integers(2) == 0

    # Greedy decoding - if we solve with it directly we return the solution
    # Do this only when we don't use rng
    if not rng_active_num:
        greedy_sol = greedy_decoding(encoded_term, decoder_s, len1, params_s, envir_s)
        valid, solution, _ = check_valid_solution(terms_to_simplify, num_terms_init, greedy_sol, envir_s, params_s,
                                                  rng_active_num, ref_term=ref_term)
        if valid:
            logger.info('Found simplified form with greedy decoding')
            logger.info('Expression found is {}'.format(solution))
            return convert_sp_forms(solution, envir_s.func_dict)

    # Beam decoding
    beam_hypothesis = generate_beam_hyp(encoded_term, len1, decoder_s, params_s)
    valid, solution = check_hypothesis(terms_to_simplify, num_terms_init, beam_hypothesis, envir_s, params_s,
                                       rng_active_num, ref_term=ref_term)
    if valid:
        logger.info('Found simplified form with beam search')
        logger.info('Expression found is {}'.format(solution))
        return convert_sp_forms(solution, envir_s.func_dict)

    # Nucleus sampling
    beam_hypothesis = generate_nucleus_hyp(encoded_term, len1, decoder_s, params_s, rng)
    valid, solution = check_hypothesis(terms_to_simplify, num_terms_init, beam_hypothesis, envir_s, params_s,
                                       rng_active_num,  ref_term=ref_term)
    if valid:
        logger.info('Found simplified form with nucleus sampling')
        logger.info('Expression found is {}'.format(solution))
        return convert_sp_forms(solution, envir_s.func_dict)

    logger.info('Could not reduce the expression' + '\n')

    return None


def single_simplification_pass(input_equation, modules, envs, params_s, rng, log_dict, inf_method=None,
                               cutoff=0.9, const_blind=False, verbose=True):
    """
    Go over all the terms in the input equation and try to find relevant terms with which they can simplify
    Once all the terms have been considered we stop the simplification search
    :param input_equation:
    :param modules:
    :param envs:
    :param params_s:
    :param cutoff:
    :param rng:
    :param log_dict:
    :param inf_method:
    :param const_blind:
    :param verbose:
    :return:
    """
    # Unpack the modules
    encoder_c, encoder_s, decoder_s = modules

    # Unpack the environments
    env_c, env_s = envs

    # Extract the numerator and denominators
    terms_num, denom = extract_num_denom(input_equation)

    # Shuffle the input terms in we use rng
    rng_active, rng_gens = rng
    if rng_active:
        rng_gens[0].shuffle(terms_num)

    # Retain variables for parsing
    terms_left = terms_num
    index_term = 0

    # Retain the added solution terms
    solution_generated = 0
    terms_simplified_num = 0
    num_simplification = 0
    short_search = False

    if verbose:
        logger.info('Start our pass over the terms in the expression')
        logger.info('We start with {} terms in the numerator'.format(len(terms_left)))
        logger.info('We start with expression {} '.format(sp.cancel(terms_left.sum()/denom)))
        logger.info('We use a similarity cut-off of {}'.format(cutoff))

    # Loop till we have considered all the terms
    while len(terms_left) > 0 and index_term < len(terms_left):

        # Find the terms most likely to cancel with the reference term
        sim_term = masked_similarity_term(env_c, terms_left[index_term], terms_left, encoder_c, const_blind=const_blind)
        simplifier_t, rest_t = find_single_simplification_terms(sim_term, terms_left, cutoff=cutoff,
                                                                denominator=denom, short_search=short_search)

        # If we expect a simplification we try it out else we consider the next term
        if simplifier_t is not None:

            num_t_attempt = len(sim_term) - len(rest_t)

            # Attempt the simplification with greedy + beam + nucleus
            if inf_method is None:
                solution_attempt = attempt_simplification(simplifier_t, encoder_s, decoder_s, env_s, params_s, rng,
                                                          const_blind=const_blind)

            # If we specify the inference methods then we only try those
            else:
                params_input = (params_s.beam_size, params_s.nucleus_p, params_s.temperature)

                # Simplify initial term and count its length
                simplifier_t = simplifier_t.cancel()
                hyps_found = all_one_shot_simplify(inf_method, env_s, (encoder_s, decoder_s), simplifier_t,
                                                   params_input, blind_const=blind_constants, rng=rng[1][1])
                solution_attempt = retain_valid_hypothesis(hyps_found, simplifier_t, rng[0])

            # If we simplified then we update the terms to consider
            if solution_attempt is not None:

                # Update the number of terms left
                terms_left = rest_t

                # For logging purposes track the simplification
                num_simplification += 1
                num_terms_simple = count_numerator_terms(solution_attempt)
                num_terms_complex = count_numerator_terms(simplifier_t)
                terms_simplified_num += num_terms_simple
                solution_generated = solution_generated + solution_attempt

                # Update the logger
                log_dict['Initial equation'].append(simplifier_t)
                log_dict['Final equation'].append(solution_attempt)
                log_dict['Initial size'].append(num_terms_complex)
                log_dict['Final size'].append(num_terms_simple)
                log_dict['Num simplifications'].append(1)
                if verbose:
                    str_search = 'short' if short_search else 'long'
                    streamlit_logger2.info('Step {}: We simplified {} terms'
                                          ' down to {}'.format(num_simplification, num_terms_complex, num_terms_simple))
                    streamlit_logger.info('We have {} terms left'
                                          ' in the numerator'.format(len(terms_left)+terms_simplified_num))
                    logger.info('We simplified down to {} terms - {} search'.format(num_terms_simple, str_search) + '\n')

                # Reset the search method and keep track of the current proposed solution
                short_search = False

            # If we only searched for a long expression we allow to search the smaller one
            elif not short_search and num_t_attempt > 4:
                short_search = True

            # If no simplification and already short search we consider the next term
            else:
                index_term += 1
                short_search = False

        else:
            index_term += 1
            short_search = False

    terms_left_end = len(terms_left)+terms_simplified_num

    if verbose:
        logger.info('Finished our pass over the terms in the expression')
        logger.info('We now have {} terms left in the numerator'.format(terms_left_end))
        streamlit_logger.info('Intermediate pass done: We have {} terms left in the numerator'.format(terms_left_end))

    # Craft back the expression from the terms left over ( maybe not as fast as could be as we will call cancel later)
    # But for now good enough - also cancel can help reduce the length of num and denom
    terms_left = terms_left.sum()
    terms_left = terms_left / denom

    # Return the expression to be considered in the next pass
    return solution_generated + terms_left, terms_left_end, num_simplification


def total_simplification(envirs, params, input_equation, modules, rng_gen, inf_method=None, const_blind=False,
                         init_cutoff=0.99, power_decay=5, verbose=True):
    """
    Given an input equation we parse through its terms as many times as possible while the
    model finds a simplified form
    :param envirs:
    :param params:
    :param input_equation:
    :param modules:
    :param rng_gen:
    :param inf_method:
    :param init_cutoff:
    :param const_blind:
    :param power_decay:
    :param dir_out:
    :param verbose:
    :return:
    """
    # Load the environment and the parameters
    envir_c, envir_s = envirs
    params_c, params_s = params

    # Initialize loop parameters
    reducing = True
    init_eq = input_equation
    rng_active = False
    len_init = count_numerator_terms(input_equation)
    len_in = len_init
    num_simplification = 0
    rng_passes = 0
    cutoff = init_cutoff
    simplification_log = {'Initial equation': [], 'Final equation': [], 'Initial size': [], 'Final size': [],
                          'Num simplifications': []}

    if verbose:
        streamlit_logger.info('Starting the simplification of {} terms !'.format(len_in))
        logger.info('\n' + 'Starting the simplification of {}'.format(input_equation))

    # Try to simplify as long as we hope for a simplification
    while reducing:

        # Do a simplification pass over all the terms in the expressions
        simple_form, len_new, num_simple = single_simplification_pass(input_equation, modules, envirs, params_s,
                                                                      (rng_active, rng_gen), simplification_log,
                                                                      inf_method=inf_method, cutoff=cutoff,
                                                                      const_blind=const_blind)

        # If the expression decreased in size or we simplified it we iterate
        if len_in > len_new or num_simple > 0:
            num_simplification += num_simple
            input_equation = simple_form
            len_in = len_new

            # Keep rng only if active and not decreasing
            # Check that we are not just cycling through terms with rng active
            rng_active = False if len_in > len_new else rng_active
            rng_passes = rng_passes + 1 if rng_active else 0
            reducing = rng_passes < 5

        # If the expression is of size 1 it is maximally simplified
        elif len_new == 1:
            if verbose:
                logger.info('Maximally simplified')
            reducing = False

        # If it has not decreased we still allow for a number of extra loops in case we need to reshuffle the expression
        # We start to randomly swap terms and simplify to expressions of same length to increase diversity
        else:
            rng_passes += 1
            if verbose:
                logger.info('No obvious simplification anymore - Using random shuffling to increase diversity')
                logger.info('Start from expression {}'.format(sp.cancel(simple_form)))
                logger.info('Start rng pass number {}'.format(rng_passes))
            rng_active = True
            reducing = rng_passes < 5

        # At each pass we lower the similarity cutoff
        cutoff = cutoff*(init_cutoff**power_decay)

    simple_form = sp.cancel(simple_form)

    if verbose:
        logger.info('Simplified form is {}'.format(simple_form))
        logger.info('Went from {} to {} terms with {} simplifications'.format(len_init, len_new, num_simplification) + '\n')
        streamlit_logger.info('Went from {} to {} terms in {} simplification steps'.format(len_init, len_new, num_simplification) + '\n')

    header_out = ['Initial equation', 'Final equation', 'Initial size', 'Final size', 'Num simplifications']

    data_out = pd.DataFrame([[init_eq, simple_form, len_init, len_new, num_simplification]], columns=header_out)
    log_frame = pd.DataFrame(simplification_log)
    data_out = pd.concat([data_out, log_frame], ignore_index=True)
    return simple_form, data_out
