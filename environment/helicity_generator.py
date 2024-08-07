"""
File containing the necessary functions for the generation of spinor helicity amplitudes
"""

import numpy as np
import random
from itertools import combinations
from environment.bracket_env import ab, sb
from add_ons.mathematica_utils import solve_diophantine_systems
from environment.utils import reorder_expr, generate_random_bk, build_scale_factor, get_expression_detail_lg_scaling
from sympy import latex, Function, sympify, fraction, cancel
from logging import getLogger
import sys
logger = getLogger()


def generate_random_bk_type(n_points, rng):
    """
    Generate the bracket type at random, along with its arguments
    :param n_points:
    :param rng:
    :return:
    """

    # Half the time generate square bracket other half is angle bracket
    bk_type = ab if rng.randint(0, 2) == 0 else sb
    return generate_random_bk(bk_type, n_points, rng)


def poisson_power(lambda_scale, rng):
    """
    Generate a power to be associated with a given bracket expression
    :param lambda_scale: expected rate of occurrence
    :param rng:
    :return:
    """

    # Draw from the poisson distribution
    result = rng.poisson(lam=lambda_scale)

    # Choose not to return 0 for the power so at minimum we return 1
    return max(1, result)


def generate_random_fraction_unbounded(l_scaling, n_points, max_terms_add, rng, zero_allowed=True):
    """
    Generate a random fraction by multiplying with random brackets
    We choose a given number of brackets for the numerator and denominator
    We also choose the power with which they should come into the expression
    :param l_scaling: expected rate of occurrence for the Poisson distribution
    :param n_points: number of external particles
    :param max_terms_add: maximum number of bracket terms in the numerator or denominator
    :param rng:
    :param zero_allowed: whether we can generate 0 as the target
    :return:
    """

    # Number of numerator bracket terms
    n_num = rng.randint(0 if zero_allowed else 1, max_terms_add + 1)

    # If we return 0 as the target amplitude
    if n_num == 0:
        return sympify(0)

    # Number of bracket denominator terms
    n_denom = rng.randint(1, max_terms_add + 1)

    return_expr = 1

    # For the numerator and denominator we generate randomly the appropriate number of bracket terms and raise
    # them to a power following the chosen Poisson distribution
    for i in range(n_num):
        return_expr *= generate_random_bk_type(n_points, rng)**poisson_power(l_scaling, rng)

    for j in range(n_denom):
        return_expr *= 1/(generate_random_bk_type(n_points, rng))**poisson_power(l_scaling, rng)

    # Randomly sample the overall sign
    sign = 1 if rng.randint(0, 2) == 0 else -1

    return return_expr * sign


def generate_random_amplitude(npt_list, rng=None, max_terms_scale=1, max_components=1, l_scale=1, str_out=False,
                              verbose=False,  info_scaling=False, session=None,
                              all_momenta=True):
    """
    Generate a random component of a tree level spinor helicity amplitude with a random number of external legs.
    We constrain the amplitude to be physically viable
    :param npt_list: list of allowed number of external particles
    :param rng: numpy random state
    :param max_terms_scale: scale factor defining the maximum number of bracket terms in the numerator or denominator
    :param max_components: maximum number of distinct numerator terms
    :param l_scale: expected rate of occurrence for the Poisson distribution
    :param str_out: whether to output the string representation of the amplitude
    :param verbose: whether to print as the generation is done
    :param info_scaling: whether to output the little group scaling information
    :param session: the mathematica session
    :param all_momenta: whether we require the generated amplitude to have all the possible external momenta labels
    :return:
    """

    # Create the random state if we don't have one
    if rng is None:
        rng = np.random.RandomState()

    # Choose the number of external particles
    n_points = rng.choice(npt_list)

    # Choose the number of distinct numerator terms
    components = 1 if max_components == 1 else rng.randint(1, max_components + 1)

    return_expr = 0
    scale_list = None

    # We start by generating a first term
    return_expr += generate_random_fraction_unbounded(l_scale, n_points, int(max_terms_scale * n_points), rng,
                                                      zero_allowed=components == 1)
    # Isolate the denominator
    denominator = fraction(return_expr)[-1]

    # Calculate the little group scaling of the expression
    scaling_list = get_expression_detail_lg_scaling(return_expr, [ab, sb], n_points)

    if info_scaling:
        scale_list = np.array(scaling_list[0]) - np.array(scaling_list[1])

    # For each additional numerator component we solve the diophantine equation to find an
    # equivalent numerator expression that we can add.
    if components > 1:
        if session is None:
            raise TypeError('Need a valid Mathematica Session to generate multiple terms and respect scaling')

        # Solve the little group scaling equation for the numerator
        coeff_add_num_list = solve_diophantine_systems(n_points, scaling_list[0], components-1, session)

        # If we are not able to solve the diophantine equation (with required number of solutions)
        if coeff_add_num_list is None:
            return None, None, None

        # Build the appropriate numerator to add from the little group scaling factors
        # Also randomly sample the overall sign and add back the common denominator
        for new_coeff_num in coeff_add_num_list:
            new_num = build_scale_factor(new_coeff_num, ab, sb, n_points)
            sign = 1 if rng.randint(0, 2) == 0 else -1
            return_expr += (sign * new_num / denominator)

        return_expr = cancel(return_expr)

    # If we are missing any external momentum in the whole expression then we try again if we require it
    if all_momenta and any([i not in np.array([list(f.args) for f in return_expr.atoms(Function)]).flatten()
                            for i in range(1, n_points+1)]):
        return generate_random_amplitude(npt_list, rng, max_terms_scale, max_components, l_scale, str_out,
                                         verbose, info_scaling, session, all_momenta)

    if verbose:
        print("Generated {}-pt amplitude".format(n_points))

    if str_out:
        return str(return_expr), n_points, scale_list
    else:
        return return_expr, n_points, scale_list
