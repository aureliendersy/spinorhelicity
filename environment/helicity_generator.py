"""
File containing the necessary functions for the generation of spinor helicity amplitudes
"""

import numpy as np
import random
from itertools import combinations
from environment.bracket_env import ab, sb
from add_ons.mathematica_utils import solve_diophantine_systems
from environment.utils import (reorder_expr, generate_random_bk, get_scaling_expr_detail, build_scale_factor,
                               get_expression_detail_lg_scaling)
from sympy import latex, Function, sympify, fraction, cancel
from logging import getLogger
import sys
logger = getLogger()


def dual_partition(nint):
    """Splits of an integer into two"""
    return [[s1 - s0 - 1 for s0, s1 in zip((-1,) + splits, splits + (nint+1,))]
            for splits in combinations(range(nint+1), 1)]


def str_bk(bk, pi, pj):
    """Return the string version of a helicity amplitude bracket"""
    return bk + '(' + str(pi) + ',' + str(pj) + ')'


def parke_taylor(bk, pi, pj, max_label):
    """Gives the Parke Taylor formula for a MHV amplitude"""
    str_denom = ''
    for i in range(1, max_label+1):
        str_denom += str_bk(bk, i, i+1 if i != max_label else 1)
        if i != max_label:
            str_denom += '*'

    return str_bk(bk, pi, pj) + '**4' + '/(' + str_denom + ')'


def generate_random_bk_type(n_points, rng, canonical=False):
    """
    Generate the bracket type at random, along with its arguments
    :param n_points:
    :param rng:
    :param canonical
    :return:
    """

    bk_type = ab if rng.randint(0, 2) == 0 else sb
    return generate_random_bk(bk_type, n_points, rng, canonical=canonical)


def generate_random_fraction(scaling, n_points, max_terms_add, rng, canonical_form):
    """
    Generate a random fraction involving momenta with the correct little group scaling
    :param scaling:
    :param n_points:
    :param max_terms_add:
    :param rng:
    :param canonical_form:
    :return:
    """

    # Randomly assign the correct skeleton
    choice_index = np.random.choice(len(dual_partition(abs(scaling))), 1)[0]
    scaling_skel = dual_partition(abs(scaling))[choice_index]

    # Select how many terms with zero little group scaling we add
    add_terms = rng.randint(0 if scaling != 0 else 1, max_terms_add + 1)

    return_expr = 1

    # Add numerator of skeleton
    for i in range(scaling_skel[0]):
        fct_temp = ab if scaling >= 0 else sb
        return_expr *= generate_random_bk(fct_temp, n_points, rng)

    # Add denominator of skeleton
    for j in range(scaling_skel[1]):
        fct_temp = sb if scaling >= 0 else ab
        return_expr *= 1/generate_random_bk(fct_temp, n_points, rng)

    # Draw the type of zero scaling term we have and add it
    for k in range(add_terms):
        term_type = rng.randint(1, 5)

        if term_type == 1:
            return_expr *= generate_random_bk(ab, n_points, rng) / generate_random_bk(ab, n_points, rng)
        elif term_type == 2:
            return_expr *= generate_random_bk(sb, n_points, rng) / generate_random_bk(sb, n_points, rng)
        elif term_type == 3:
            return_expr *= generate_random_bk(sb, n_points, rng) * generate_random_bk(ab, n_points, rng)
        elif term_type == 4:
            return_expr *= 1/(generate_random_bk(sb, n_points, rng) * generate_random_bk(ab, n_points, rng))

    if canonical_form:
        return_expr = reorder_expr(return_expr)

    return return_expr


def poisson_power(lambda_scale, rng):
    """
    Generate a power to be associated with a given bracket expression
    :param lambda_scale:
    :return:
    """

    result = rng.poisson(lam=lambda_scale)

    return max(1, result)


def generate_random_fraction_unbounded(l_scaling, n_points, max_terms_add, rng, canonical_form, zero_allowed=True):
    """
    Generate a random fraction by multiplying with random brackets
    We choose a given number of brackets for the numerator and denominator
    We also choose the power with which they should come into the expression
    :param l_scaling:
    :param n_points:
    :param max_terms_add:
    :param rng:
    :param canonical_form:
    :param zero_allowed
    :return:
    """

    n_num = rng.randint(0 if zero_allowed else 1, max_terms_add + 1)

    if n_num == 0:
        return sympify(0)

    n_denom = rng.randint(1, max_terms_add + 1)

    return_expr = 1

    for i in range(n_num):
        return_expr *= generate_random_bk_type(n_points, rng, canonical=canonical_form)**poisson_power(l_scaling, rng)

    for j in range(n_denom):
        return_expr *= 1/(generate_random_bk_type(n_points, rng,
                                                  canonical=canonical_form))**poisson_power(l_scaling, rng)

    sign = 1 if rng.randint(0, 2) == 0 else -1

    return return_expr * sign


def generate_random_amplitude(npt_list, rng=None, max_terms_scale=1, max_components=1, l_scale=1, str_out=False,
                              verbose=False, canonical_form=False, generator_id=1, info_scaling=False, session=None,
                              all_momenta=True):
    """
    Generate a random component of a tree level spinor helicity amplitude with a random number of external legs.
    We constrain the amplitude to be physically viable
    :param npt_list:
    :param rng:
    :param max_terms_scale:
    :param max_components:
    :param l_scale:
    :param str_out:
    :param verbose:
    :param canonical_form:
    :param generator_id:
    :param info_scaling:
    :param session:
    :param all_momenta:
    :return:
    """

    if rng is None:
        rng = np.random.RandomState()
    n_points = rng.choice(npt_list)

    if generator_id == 1:
        n_pos_h = 2 if n_points == 4 else rng.randint(2, n_points-1)
        n_neg_h = n_points-n_pos_h

    components = 1 if max_components == 1 else rng.randint(1, max_components + 1)

    return_expr = 0
    scale_list = None

    # We start by generating a first overall skeleton
    if generator_id == 1:
        return_expr += generate_random_fraction(-n_pos_h + n_neg_h, n_points, int(max_terms_scale * n_points), rng,
                                                canonical_form=canonical_form)
    else:
        return_expr += generate_random_fraction_unbounded(l_scale, n_points, int(max_terms_scale * n_points),
                                                          rng, canonical_form=canonical_form,
                                                          zero_allowed=components == 1)
    denominator = fraction(return_expr)[-1]
    scaling_list = get_expression_detail_lg_scaling(return_expr, [ab, sb], n_points)
    if info_scaling:
        scale_list = np.array(scaling_list[0]) - np.array(scaling_list[1])

    # For each additional component we only modify the numerators
    # We solve the diophantine equation to find an equivalent expression that we can add

    if components > 1:
        if session is None:
            raise TypeError('Need a valid Mathematica Session to generate multiple terms and respect scaling')
        coeff_add_num_list = solve_diophantine_systems(n_points, scaling_list[0], components-1, session)
        # If we are not able to solve the diophantine equation (with required number of solutions)
        if coeff_add_num_list is None:
            return None, None, None
        for new_coeff_num in coeff_add_num_list:
            new_num = build_scale_factor(new_coeff_num, ab, sb, n_points)
            sign = 1 if rng.randint(0, 2) == 0 else -1
            return_expr += (sign * new_num / denominator)

        return_expr = cancel(return_expr)

    # If we are missing any external momentum in the whole expression then we try again
    if all_momenta and any([i not in np.array([list(f.args) for f in return_expr.atoms(Function)]).flatten()
                            for i in range(1, n_points+1)]):
        return generate_random_amplitude(npt_list, rng, max_terms_scale, max_components, l_scale, str_out,
                                         verbose, canonical_form, generator_id, info_scaling, session, all_momenta)

    if verbose:
        print("Generated {}-pt amplitude with {} positive polarizations".format(n_points, n_pos_h))

    if str_out:
        return str(return_expr), n_points, scale_list
    else:
        return return_expr, n_points, scale_list
