"""
Methods relevant for simplifying a large spinor helicity amplitude using the contrastive model
"""

from logging import getLogger
import numpy as np
import pandas as pd
import sympy as sp
from sympy import latex
import time

from environment.utils import to_cuda
from add_ons.mathematica_utils import sp_to_mma
from model.simplifier_methods import (blind_constants, extract_num_denom, all_one_shot_simplify, fast_one_shot_simplify,
                                      retain_valid_hypothesis, count_numerator_terms)
import torch
import torch.nn as nn

# Define two separate logger for the streamlit application
streamlit_logger = getLogger(__name__)
streamlit_logger2 = getLogger(__name__+'2')


def encode_term(envir_c, term, encoder_c):
    """
    Given an input sympy term we convert it to prefix notation and pass it through the
    encoder network of the contrastive learning setup
    :param envir_c: contrastive environment
    :param term: input sympy term
    :param encoder_c: embedding contrastive network
    :return:
    """

    # Convert the sympy input to prefix then to the respective ids
    t_prefix = envir_c.sympy_to_prefix(term)
    t_in = to_cuda(torch.LongTensor([envir_c.eos_index] + [envir_c.word2id[w] for w in t_prefix] +
                                    [envir_c.eos_index]).view(-1, 1))[0]
    len_in = to_cuda(torch.LongTensor([len(t_in)]))[0]

    # Forward of the encoder
    with torch.no_grad():
        encoded = encoder_c('fwd', x=t_in, lengths=len_in, causal=False)

    return encoded


def normalize_term(term_in):
    """
    Given a term in the numerator return the same term normalized (constant 1 or -1)
    :param term_in: sympy term to be normalized
    :return:
    """

    # If we just have a number we give +-1 back
    if isinstance(term_in, sp.Integer):
        return term_in/abs(term_in)

    # If we get a bracket functional raised to some power it is already normalized
    if isinstance(term_in, sp.Function) or isinstance(term_in, sp.Pow):
        return term_in

    if not isinstance(term_in, sp.Mul):
        raise ValueError('Expected a multiplicative term to normalize for {}'.format(term_in))

    # Extract the constants in front of each numerator term
    const_vect = [term_mult for term_mult in term_in.args if isinstance(term_mult, sp.Integer)]

    # Sanity check
    if len(const_vect) > 1:
        raise ValueError('Found two constants in a numerator term when normalizing')

    const = 1 if len(const_vect) == 0 else abs(const_vect[0])

    return term_in / const


def masked_similarity_term(envir_c, term_ref, terms_comp, encoder_c, const_blind=False, masked=True):
    """
    Given a reference term we compute its cosine similarity with a list of target terms.
    We calculate the cosine similarity only on the parts of the terms that do not
    share similar factors in common
    :param envir_c: contrastive environment
    :param term_ref: reference numerator term we compute the cosine similarity on
    :param terms_comp: input vector of numerator terms
    :param encoder_c: embedding contrastive network
    :param const_blind: whether we blind to constants before computing the cosine similarity
    :param masked: whether we mask similar factors before computing the cosine similarity
    :return:
    """

    # Define the cosine similarity metric
    metric_sim = nn.CosineSimilarity(dim=-1)
    similarity_vect = torch.zeros(len(terms_comp))

    # Compute the metric for each term that is in the input vector
    for i, term in enumerate(terms_comp):

        # If we mask then we cancel all common factors
        if masked:
            newterm_ref, newterm_comp = sp.fraction(sp.cancel(term_ref / term))
        else:
            newterm_ref, newterm_comp = term_ref, term

        # When blinding constants we have to normalize the terms (still stay sensitive to sign)
        if const_blind:
            newterm_ref = normalize_term(newterm_ref)
            newterm_comp = normalize_term(newterm_comp)

        # Pass the terms through the embedding network
        encoded_ref = encode_term(envir_c, newterm_ref, encoder_c)
        encoded_comp = encode_term(envir_c, newterm_comp, encoder_c)

        # Compute the cosine similarity
        similarity_vect[i] = metric_sim(encoded_ref, encoded_comp)

    return similarity_vect


def find_single_simplification_terms(similarity_vect, terms_comp, cutoff=0.9, denominator=None, short_search=False):
    """
    Given a similarity vector we retain the expression that holds the greatest promise in simplifying
    :param similarity_vect: vector of cosine similarities
    :param terms_comp: vector of numerator terms
    :param cutoff: cosine similarity cutoff for forming groups
    :param denominator: denominator of the overall amplitude
    :param short_search: whether to form a group of size 4 at most
    :return:
    """

    # Retain the similarities that are above the cutoff
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
    terms_to_simplify = relevant_terms.sum()

    # Also return the other non-combined terms
    rest_terms = terms_comp[~mask]

    # Add back the overall denominator
    if denominator is not None:
        terms_to_simplify = terms_to_simplify / denominator

    return terms_to_simplify, rest_terms


def single_simplification_pass(input_equation, modules, envs, params_s, rng, log_dict, cache, inf_method=None,
                               cutoff=0.9, const_blind=False, fast_inf=False, verbose=True):
    """
    Go over all the terms in the input equation and try to find relevant terms with which they can simplify
    Once all the terms have been considered we stop the simplification search
    :param input_equation: input sympy amplitude
    :param modules: tuple of embedding network and encoder-decoder simplifier models
    :param envs: tuple of contrastive and simplifier environments
    :param params_s: parameters of the simplifier environment
    :param cutoff: cutoff for the similarity grouping
    :param rng: tuple of (whether rng is active, rng generators)
    :param log_dict: logging dictionary
    :param cache: cache of unsuccessful simplification steps
    :param inf_method: list of inference methods to use
    :param const_blind: whether to perform the simplification blind to constants
    :param fast_inf: whether to retain the first valid solution that is found
    :param verbose: whether to output more information to the logger
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

    # Loop till we have considered all the terms
    while len(terms_left) > 0 and index_term < len(terms_left):

        # Find the terms most likely to cancel with the reference term
        sim_term = masked_similarity_term(env_c, terms_left[index_term], terms_left, encoder_c, const_blind=const_blind)
        simplifier_t, rest_t = find_single_simplification_terms(sim_term, terms_left, cutoff=cutoff,
                                                                denominator=denom, short_search=short_search)

        # If we expect a simplification we try it out else we consider the next term
        if simplifier_t is not None:

            # Simplify initial term and count its length
            simplifier_t = simplifier_t.cancel()
            num_t_attempt = len(sim_term) - len(rest_t)

            # If we specify the inference methods then we only try those
            # Don't look at the cache if we are doing random shuffling
            if simplifier_t not in cache or rng_active:
                params_input = (params_s.beam_size, params_s.nucleus_p, params_s.temperature)

                # Do a slow inference (look through all methods and pick the best solution)
                if not fast_inf:
                    hyps_found = all_one_shot_simplify(inf_method, env_s, (encoder_s, decoder_s), simplifier_t,
                                                       params_input, blind_const=blind_constants, rng=rng[1][1])
                    solution_attempt = retain_valid_hypothesis(hyps_found, simplifier_t, rng[0])

                # Do a fast inference (return the first valid solution without doing all inference methods)
                else:
                    solution_attempt = fast_one_shot_simplify(inf_method, env_s, (encoder_s, decoder_s), simplifier_t,
                                                              params_input, rng[0], blind_const=blind_constants,
                                                              rng=rng[1][1])
            else:
                solution_attempt = None

            # If we simplified then we update the terms to consider
            if solution_attempt is not None and solution_attempt != 'Equiv':

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
                    streamlit_logger2.info('Step {}: We simplified {} terms'
                                          ' down to {}'.format(num_simplification, num_terms_complex, num_terms_simple))
                    streamlit_logger.info('We have {} terms left'
                                          ' in the numerator'.format(len(terms_left)+terms_simplified_num))

                # Reset the search method and keep track of the current proposed solution
                short_search = False

            # If we only searched for a long expression we allow to search the smaller one
            elif not short_search and num_t_attempt > 4:
                short_search = True

                # Update the failed caches (don't update if the solution was equivalent in complexity)
                if simplifier_t not in cache and solution_attempt != 'Equiv':
                    cache.append(simplifier_t)

            # If no simplification and already short search we consider the next term
            else:
                index_term += 1
                short_search = False

                # Update the failed caches (don't update if the solution was equivalent in complexity)
                if simplifier_t not in cache and solution_attempt != 'Equiv':
                    cache.append(simplifier_t)

        # When we expect no simplification we move to the next term
        else:
            index_term += 1
            short_search = False

    terms_left_end = len(terms_left)+terms_simplified_num

    if verbose:
        streamlit_logger.info('Intermediate pass done: We have {} terms left in the numerator'.format(terms_left_end))

    # Craft back the expression from the terms left over ( maybe not as fast as could be as we will call cancel later)
    # But for now good enough - also cancel can help reduce the length of num and denom
    terms_left = terms_left.sum()
    terms_left = terms_left / denom

    # Return the expression to be considered in the next pass
    return solution_generated + terms_left, terms_left_end, num_simplification


def total_simplification(envirs, params, input_equation, modules, rng_gen, inf_method=None, const_blind=False,
                         init_cutoff=0.99, power_decay=0.5, fast_inf=False, verbose=True):
    """
    Given an input equation we parse through its terms as many times as possible while the
    model finds a simplified form
    :param envirs: tuple of contrastive and simplifier environments
    :param params: tuple of contrastive and simplifier parameters
    :param input_equation: input sympy amplitude
    :param modules: tuple of embedding network and encoder-decoder simplifier models
    :param rng_gen: tuple of (rng numpy , rng torch) generators
    :param inf_method: list of inference methods to use
    :param init_cutoff: initial similarity cutoff (c0)
    :param const_blind: whether to perform the simplification blind to constants
    :param power_decay: power decay factor to the similarity cutoff (alpha)
    :param fast_inf: whether to retain the first valid solution that is found
    :param verbose: whether to output more information to the logger
    :return:
    """

    # Start the timer
    start_time = time.time()

    # Load the environment and the parameters
    params_c, params_s = params

    # Initialize loop parameters
    reducing = True  # Can we still reduce the equation
    rng_active = False  # Do we allow for random shuffling of the expression
    num_simplification = 0
    rng_passes = 0
    cutoff = init_cutoff
    cache_invalid = []

    # Load the equation and record the number of terms in the numerator
    init_eq = input_equation
    len_init = count_numerator_terms(input_equation)
    len_in = len_init

    # Initialize a logger for all simplification steps taken
    simplification_log = {'Initial equation': [], 'Final equation': [], 'Initial size': [], 'Final size': [],
                          'Num simplifications': []}

    if verbose:
        streamlit_logger.info('Starting the simplification of {} terms !'.format(len_in))

    # Try to simplify as long as we hope for a simplification
    while reducing:

        # Do a simplification pass over all the terms in the expressions
        simple_form, len_new, num_simple = single_simplification_pass(input_equation, modules, envirs, params_s,
                                                                      (rng_active, rng_gen), simplification_log,
                                                                      cache_invalid, inf_method=inf_method,
                                                                      cutoff=cutoff, const_blind=const_blind,
                                                                      fast_inf=fast_inf, verbose=verbose)

        # If the expression decreased in size or we simplified it we iterate
        if len_in > len_new or num_simple > 0:
            num_simplification += num_simple
            input_equation = simple_form
            len_in = len_new

            # Keep rng only if active and not decreasing
            # Check that we are not just cycling through terms with rng active
            rng_active = False if len_in > len_new else rng_active
            rng_passes = rng_passes + 1 if rng_active else 0
            if rng_active:
                streamlit_logger.info('{} terms: start shuffling pass {}/5'.format(len_new, rng_passes))
            reducing = rng_passes < 5

        # If the expression is of size 1 it is maximally simplified
        elif len_new == 1:
            reducing = False

        # If it has not decreased we still allow for a number of extra loops in case we need to reshuffle the expression
        # We start to randomly swap terms and simplify to expressions of same length to increase diversity
        else:
            rng_passes += 1
            if verbose:
                streamlit_logger.info('No obvious simplification at {} terms:'
                                      ' start shuffling pass {}/5'.format(len_new, rng_passes))
            rng_active = True
            reducing = rng_passes < 5

        # At each pass we lower the similarity cutoff
        cutoff = cutoff*(init_cutoff**power_decay)

    # Return the final expression in a minimal form
    simple_form = sp.cancel(simple_form)
    exec_time = time.time() - start_time
    if verbose:
        streamlit_logger.info('Went from {} to {} terms in {} simplification'
                              ' steps and {:.1f} seconds'.format(len_init, len_new, num_simplification, exec_time) + '\n')

    # Prepare the output dataframe that records each simplification step
    header_out = ['Initial equation', 'Final equation', 'Initial size', 'Final size', 'Num simplifications']
    data_out = pd.DataFrame([[init_eq, simple_form, len_init, len_new, num_simplification]], columns=header_out)
    log_frame = pd.DataFrame(simplification_log)
    data_out = pd.concat([data_out, log_frame], ignore_index=True)
    data_out['Time Taken'] = exec_time

    # Add different output formats
    data_out["Initial equation Latex"] = data_out["Initial equation"].apply(latex)
    data_out["Final equation Latex"] = data_out["Final equation"].apply(latex)
    data_out["Initial equation S@M"] = data_out["Initial equation"].apply(
        sp_to_mma, args=(envirs[-1].npt_list, envirs[-1].func_dict))
    data_out["Final equation S@M"] = data_out["Final equation"].apply(
        sp_to_mma, args=(envirs[-1].npt_list, envirs[-1].func_dict))
    data_out['Initial equation Mathematica'] = data_out['Initial equation S@M'].apply(
        lambda x: x.replace("Spaa", "ab").replace("Spbb", "sb"))
    data_out['Final equation Mathematica'] = data_out['Final equation S@M'].apply(
        lambda x: x.replace("Spaa", "ab").replace("Spbb", "sb"))

    return simple_form, data_out
