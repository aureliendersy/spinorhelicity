"""
File containing the necessary functions for the generation of spinor helicity amplitudes
"""

import numpy as np
import random
import sympy as sp
from itertools import combinations
from spin_helicity_env import ab, sb, SpinHelExpr
from sympy import latex, Function


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


def generate_random_bk(bk_fct, n_points):
    """Provided with the bracket type, generate a bracket with random momenta"""
    pi = random.randint(1, n_points)
    pj = random.choice([i for i in range(1, n_points + 1) if i not in [pi]])
    return bk_fct(pi, pj)


def generate_random_fraction(scaling, n_points, max_terms_add):
    """Generate a random fraction involving momenta with the correct little group scaling"""

    # Randomly assign the correct skeleton
    scaling_skel = random.choice(dual_partition(abs(scaling)))

    # Select how many terms with zero little group scaling we add
    add_terms = random.randint(0 if scaling != 0 else 1, max_terms_add)

    return_expr = 1

    # Add numerator of skeleton
    for i in range(scaling_skel[0]):
        fct_temp = ab if scaling >= 0 else sb
        return_expr *= generate_random_bk(fct_temp, n_points)

    # Add denominator of skeleton
    for j in range(scaling_skel[1]):
        fct_temp = sb if scaling >= 0 else ab
        return_expr *= 1/generate_random_bk(fct_temp, n_points)

    # Draw the type of zero scaling term we have and add it
    for k in range(add_terms):
        term_type = random.randint(1, 4)

        if term_type == 1:
            return_expr *= generate_random_bk(ab, n_points) / generate_random_bk(ab, n_points)
        elif term_type == 2:
            return_expr *= generate_random_bk(sb, n_points) / generate_random_bk(sb, n_points)
        elif term_type == 3:
            return_expr *= generate_random_bk(sb, n_points) * generate_random_bk(ab, n_points)
        elif term_type == 4:
            return_expr *= 1/(generate_random_bk(sb, n_points) * generate_random_bk(ab, n_points))

    return return_expr


def generate_random_amplitude(max_n_points, max_terms_scale=1, max_components=1, gluon_only=False, str_out=False):
    """Generate a random component of a tree level spinor helicity amplitude with a random number of external legs.
    We constrain the amplitude to be physically viable"""

    n_points = random.randint(4, max_n_points)

    n_pos_h = random.randint(2, n_points-2)
    n_neg_h = n_points-n_pos_h

    components = random.randint(1, max_components)

    # Use the Parke Taylor formula if we are interested in amplitudes involving only gluons
    if (n_pos_h == 2 or n_neg_h == 2) and gluon_only and str_out:
        bk = 'ab' if random.randint(0, 1) == 1 else 'sb'
        pi = random.randint(1, max_n_points)
        pj = random.choice([i for i in range(1, max_n_points+1) if i not in [pi]])
        return parke_taylor(bk, pi, pj, n_points)

    return_expr = 0

    # For each individual fraction we generate a new expression. We define the number of legs and little group scaling
    for i in range(components):
        return_expr += generate_random_fraction(-n_pos_h+n_neg_h, n_points, int(max_terms_scale*n_points))

    # If we are missing any external momentum in the whole expression then we try again
    if any([i not in np.array([list(f.args) for f in return_expr.atoms(Function)]).flatten()
            for i in range(1, n_points+1)]):
        return generate_random_amplitude(max_n_points, max_terms_scale, max_components, gluon_only, str_out)

    if str_out:
        return str(return_expr)
    else:
        return return_expr


if __name__ == '__main__':
    from sympy import latex
    expr1 = SpinHelExpr(generate_random_amplitude(5, str_out=True))
    print(expr1)
    print(latex(expr1.sp_expr))
    print(latex(expr1))
