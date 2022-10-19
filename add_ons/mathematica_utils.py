"""
Set of routines used to make the connection with the mathematica Kernel
"""

import sympy as sp
import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
from logging import getLogger

logger = getLogger()

ZERO_ERROR_POW = 9


def start_wolfram_session(kernel_path=None, sm_package=True, lib_path=None):
    """Start a Wolfram session and return it"""

    if kernel_path is None:
        session = WolframLanguageSession()
    else:
        session = WolframLanguageSession(kernel=kernel_path)
    session.start()

    if sm_package:
        if lib_path is not None:
            session.evaluate(wlexpr('$SpinorsPath = "{}"'.format(lib_path)))
            session.evaluate(wlexpr('Get[ToFileName[{$SpinorsPath}, "Spinors.m"]]'))
            # poly_path = session.evaluate(wl.SetDirectory(lib_path))
        else:
            pass
    return session


def end_wolfram_session(session):
    """Terminate session"""
    session.terminate()


def declare_spinors(session, npt, verbose=True):
    """
    Use the Wolfram session to declare a list of spinor to be used by the S@M package
    :param session:
    :param npt:
    :param verbose:
    :return:
    """
    if verbose:
        logger.info("Declaring spinors for {}-pt amplitudes".format(npt))

    spinor_list = ''
    for i in range(1, npt+1):
        spinor_list += 'a{}{},'.format(npt, i)
    spinor_list = spinor_list[:-1]

    session.evaluate(wlexpr('DeclareSpinor[{}]'.format(spinor_list)))

    return spinor_list


def generate_random_momenta(session, npt, verbose=True):
    """
    Generate a set of on-shell momenta that sums up to zero.
    Will be used for a numerical evaluation
    :param session:
    :param npt:
    :param verbose:
    :return:
    """
    # Start by declaring the spinors if it has not been done already
    spinor_list = declare_spinors(session, npt, verbose=verbose)

    session.evaluate(wlexpr('GenMomenta[{{{}}}]'.format(spinor_list)))

    if verbose:
        logger.info('Declare momenta for the {}-pt amplitude at the numerical values'.format(npt))

        for i in range(1, npt+1):
            logger.info(session.evaluate(wlexpr('Num4V[a{}{}]'.format(npt, i))))


def initialize_numerical_check(npt_max, kernel_path=None, sm_package=True, lib_path=None, verbose=True):
    """
    Initialize the wolfram environment suitable for conducting a numerical check
    :param npt_max:
    :param kernel_path:
    :param sm_package:
    :param lib_path:
    :param verbose:
    :return:
    """
    session = start_wolfram_session(kernel_path=kernel_path, sm_package=sm_package, lib_path=lib_path)

    for i in range(4, npt_max+1):
        generate_random_momenta(session, i, verbose=verbose)

    return session


def sp_to_mma(sp_expr, bracket_tokens=False, func_dict=None):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    :param sp_expr:
    :param bracket_tokens:
    :param func_dict:
    :return:
    """
    if not bracket_tokens:
        return sp_to_mma_single_token(sp_expr)
    else:
        if func_dict is None:
            raise AttributeError("Need the function dictionnary to evaluate with bracket tokens")
        return sp_to_mma_bracket_token(sp_expr, func_dict)


def sp_to_mma_bracket_token(sp_expr, func_dict):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    Assumes that the relevant variables have been previously initialized
    Assumes that each bracket is a single token
    :param sp_expr:
    :param func_dict:
    :return:
    """
    brackets = list(sp_expr.free_symbols)
    momentum_set = set([int(bracket.name[-1]) for bracket in brackets] + [int(bracket.name[-2]) for bracket in brackets])
    n_dep = len(momentum_set)
    npt = max(momentum_set)

    # If we did not get enough momenta show it
    if n_dep < 4:
        logger.error("Got an expression which depends on less than 4 momenta")

    args_npt = [sp.Symbol('a{}{}'.format(npt, i)) for i in range(1, npt + 1)]

    dict_replace = {}
    for bracket in brackets:
        name_bk = bracket.name[0:2]
        rule = func_dict[name_bk](args_npt[int(bracket.name[-2]) - 1], args_npt[int(bracket.name[-1]) - 1])
        dict_replace.update({bracket: rule})

    sp_expr = sp_expr.subs(dict_replace)

    mma_str = sp.mathematica_code(sp_expr)
    mma_str = mma_str.replace('sb', 'Spbb').replace('ab', 'Spaa')

    return mma_str


def sp_to_mma_single_token(sp_expr):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    Assumes that the relevant variables have been previously initialized
    Assumes that each token corresponds to a single word
    :param sp_expr:
    :return:
    """

    func_list = list(sp_expr.atoms(sp.Function))
    momentum_set = set(sum([func.args for func in func_list], ()))
    n_dep = len(momentum_set)
    npt = max([int(momentum.name[-1]) for momentum in list(momentum_set)])

    # If we did not get enough momenta show it
    if n_dep < 4:
        logger.error(str(sp_expr))
        logger.error("Got an expression which depends on less than 4 momenta")

    args_npt = [sp.Symbol('a{}{}'.format(npt, i)) for i in range(1, npt + 1)]

    replace_dict_var = {sp.Symbol('p{}'.format(i)): args_npt[i - 1] for i in range(1, npt + 1)}

    sp_expr = sp_expr.subs(replace_dict_var)

    mma_str = sp.mathematica_code(sp_expr)
    mma_str = mma_str.replace('sb', 'Spbb').replace('ab', 'Spaa')

    return mma_str


def check_numerical_equiv(session, mma_hyp, mma_tgt):
    """
    Check the numerical equivalence between the hypothesis and the target
    :param session:
    :param mma_hyp:
    :param mma_tgt:
    :return:
    """

    res_diff = session.evaluate(wlexpr('Abs[N[(({})-({}))]]'.format(mma_hyp, mma_tgt)))

    try:
        valid = res_diff < 10**(-ZERO_ERROR_POW)
    except:
        return False, res_diff

    if not valid:
        res_add = session.evaluate(wlexpr('Abs[N[(({})+({}))]]'.format(mma_hyp, mma_tgt)))
        if res_add < 10**(-ZERO_ERROR_POW):
            logger.info("We got the wrong overall sign")

    return valid, res_diff
