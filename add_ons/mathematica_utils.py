"""
Set of routines used to make the connection with the mathematica Kernel
"""

import sympy as sp
import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
from logging import getLogger
from diophantine import solve as diophantine_solve
from sympy import latex
import random
import re
import pandas as pd

logger = getLogger()

ZERO_ERROR_POW = 9


def start_wolfram_session(kernel_path=None, sm_package=True, lib_path=None):
    """Start a Wolfram session and return it"""

    # Point towards the Mathematica Kernel
    if kernel_path is None:
        session = WolframLanguageSession()
    else:
        session = WolframLanguageSession(kernel=kernel_path)
    session.start()

    # If we load the S@M Package we also point to it
    if sm_package:
        if lib_path is not None:
            session.evaluate(wlexpr('$SpinorsPath = "{}"'.format(lib_path)))
            session.evaluate(wlexpr('Get[ToFileName[{$SpinorsPath}, "Spinors.m"]]'))
        else:
            pass
    return session


def end_wolfram_session(session):
    """Terminate session"""
    session.terminate()


def declare_spinors(session, npt, verbose=True):
    """
    Use the Wolfram session to declare a list of spinor to be used by the S@M package
    :param session: mathematica session
    :param npt: number of external particles
    :param verbose:
    :return:
    """
    if verbose:
        logger.info("Declaring spinors for {}-pt amplitudes".format(npt))

    spinor_list = ''
    # Declare a list of appropriate momenta labels
    for i in range(1, npt+1):
        spinor_list += 'a{}{},'.format(npt, i)
    spinor_list = spinor_list[:-1]

    session.evaluate(wlexpr('DeclareSpinor[{}]'.format(spinor_list)))

    return spinor_list


def generate_random_momenta(session, npt, verbose=True):
    """
    Generate a set of on-shell momenta that sums up to zero.
    Will be used for a numerical evaluation
    :param session: mathematica session
    :param npt: number of external particles
    :param verbose:
    :return:
    """
    # Start by declaring the spinors
    spinor_list = declare_spinors(session, npt, verbose=verbose)

    # Generate a set of valid momenta (light-like and summing to 0)
    session.evaluate(wlexpr('GenMomenta[{{{}}}]'.format(spinor_list)))

    if verbose:
        logger.info('Declare momenta for the {}-pt amplitude at the numerical values'.format(npt))

        for i in range(1, npt+1):
            logger.info(session.evaluate(wlexpr('Num4V[a{}{}]'.format(npt, i))))


def initialize_numerical_check(npt_max, kernel_path=None, sm_package=True, lib_path=None, verbose=True):
    """
    Initialize the wolfram environment suitable for conducting a numerical check
    :param npt_max: maximum number of external particles
    :param kernel_path: path to the mathematica kernel
    :param sm_package: Whether we use the S@M library
    :param lib_path: Path to the S@M library
    :param verbose:
    :return:
    """
    session = start_wolfram_session(kernel_path=kernel_path, sm_package=sm_package, lib_path=lib_path)

    for i in range(4, npt_max+1):
        generate_random_momenta(session, i, verbose=verbose)

    return session


def initialize_solver_session(kernel_path=None):
    session = start_wolfram_session(kernel_path=kernel_path)
    return session


def solve_diophantine_systems(n_points, coefficients, num_sol, session):
    """
    If we want to get different solutions to the same diophantine equation
    :param n_points: number of external particles
    :param coefficients: array of coefficients (position 0 is mass dimension and the rest is little group scaling)
    :param num_sol: the number of different solutions that we want to recover
    :param session: mathematica session
    :return:
    """

    # Check if we have the correct number of coefficients fed
    assert len(coefficients) == n_points + 1

    eqvar = ["a{}{}".format(i, j) for i in range(1, n_points) for j in range(i + 1, n_points + 1)] \
            + ["b{}{}".format(i, j) for i in range(1, n_points) for j in range(i + 1, n_points + 1)]

    # For small enough mass dimensions we sample random solutions
    if abs(coefficients[0]) < 10:
        solutions = [solve_diophantine_system_mma(coefficients, session, eqvar) for i in range(num_sol)]

    # If the mass dimension is too large then return the first solution
    # To get additional solutions we have to explicitly forbid them
    if abs(coefficients[0]) >= 10:
        solutions = []
        for i in range(num_sol):
            solutions.append(solve_diophantine_system_mma(coefficients, session, eqvar, solutions))

    # If we did not find the appropriate number of solutions that we return None
    if None in solutions:
        return None
    else:
        return solutions


def solve_diophantine_system(n_points, coefficients, session):
    """
    To solve a single diophantine equation, potentially allowing for non-negative solutions
    Only correct for the scaling of numerator like terms
    :param n_points: number of external particles
    :param coefficients: array of coefficients (position 0 is mass dimension and the rest is little group scaling)
    :param session: mathematica session
    :return:
    """

    # Check if we have the correct number of coefficients fed
    assert len(coefficients) == n_points + 1

    eqvar = ["a{}{}".format(i, j) for i in range(1, n_points) for j in range(i+1, n_points+1)]\
            + ["b{}{}".format(i, j) for i in range(1, n_points) for j in range(i + 1, n_points + 1)]

    # If we have to solve a trivial system
    if all(coeffs == 0 for coeffs in coefficients):
        return [0 for i in range(len(eqvar))]

    if coefficients[0] == 0 and any(coeff > 0 for coeff in coefficients[1:]):
        return None

    # Can ask to use the python solution by default
    if session == 'NotRequired':
        return solve_diophantine_system_python(coefficients, eqvar)
    else:
        # Try to solve with MMA (correct only for the numerator scalings)
        # If not possible then solve locally (with minimal solution set that also modifies the denominators)
        sol = solve_diophantine_system_mma(coefficients, session, eqvar)
        if sol is None:
            return solve_diophantine_system_python(coefficients, eqvar)
        else:
            return sol


def solve_diophantine_system_python(coefficients, eqvar):
    """
    Use the diophantine library to find the solution with the least square coefficients
    :param coefficients: array of coefficients (position 0 is mass dimension and the rest is little group scaling)
    :param eqvar: list of variables (coefficients in front of square and angle brackets)
    :return:
    """

    # Each bracket has a mass dimension of 1
    mass_dim = [1 for i in range(len(eqvar))]

    # The little group scaling is +-1 depending on the nature of the bracket (square or angle)
    lg_eqs = [list(np.array([1 if 'a' in vara and str(i+1) in vara else 0 for vara in eqvar]) + np.array([-1 if 'b' in varb and str(i+1) in varb else 0 for varb in eqvar])) for i in range(len(coefficients[1:]))]

    # Construct the appropriate matrices defining the diophantine equation
    a_matrix = sp.Matrix([mass_dim] + lg_eqs)
    b_matrix = sp.Matrix(coefficients)

    # Return a random small solution to the diophantine equation
    sol_random = random.choice(diophantine_solve(a_matrix, b_matrix))

    return [sol_random[i, 0] for i in range(len(eqvar))]


def solve_diophantine_system_mma(coefficients, session, eqvar, prev_sol=None):
    """
    Call the Mathematica Kernel to solve the equation
    Coefficients are of the form (mass dimension, Little group scaling coefficients)
    :param coefficients: array of coefficients (position 0 is mass dimension and the rest is little group scaling)
    :param session: mathematica session
    :param eqvar: list of variables (coefficients in front of square and angle brackets)
    :param prev_sol: solution vectors to the diophantine equation
    :return:
    """

    # Construct the equation system in  Mathematica language
    eqvarstr = '{' + ','.join(eqvar) + '}'
    mass_dim_eq = '+'.join(eqvar) + '==' + str(coefficients[0])
    lg_eqs = ['+'.join([vara for vara in eqvar if 'a' in vara and str(i+1) in vara]) + '-' + '-'.join([varb for varb in eqvar if 'b' in varb and str(i+1) in varb]) + '==' + str(coeff) for i, coeff in enumerate(coefficients[1:])]

    systemstr = '{' + ','.join([mass_dim_eq] + lg_eqs) + '}'

    if abs(coefficients[0]) < 10:
        # For a truly random solution of the equation. Not too expensive if we don't have a lot of brackets
        commandstr = 'Check[RandomChoice[Solve[{}, {}, NonNegativeIntegers]][[;; , -1]],"No"]//Quiet'.format(systemstr, eqvarstr)

    else:
        # For a fast solution. Not always random anymore but necessary if we have to solve a system with too many sols
        # If we have access to previous solutions we can input them there to ensure that we dont get back the same thing
        if prev_sol is not None and len(prev_sol) > 0:
            add_str = ','.join([' || '.join(eqvar[i] + '!={}'.format(sol) for i, sol in enumerate(prev))
                                for prev in prev_sol])
            systemstr = systemstr[:-1] + ',' + add_str + '}'
        commandstr = 'Check[FindInstance[{}, {}, NonNegativeIntegers, RandomSeeding -> Automatic][[1, ;; , -1]],"No"]//Quiet'.format(systemstr, eqvarstr)

    # Run the Mathematica command via the Mathematica session
    if session is None:
        session = initialize_solver_session()
        resultsystem = session.evaluate(wlexpr(commandstr))
        end_wolfram_session(session)
    else:
        resultsystem = session.evaluate(wlexpr(commandstr))

    # If we find no solution
    if resultsystem == "No":
        return None

    return resultsystem


def sp_to_mma(sp_expr, npts_list, bracket_tokens=False, func_dict=None):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    :param sp_expr: amplitude expression in sympy format
    :param npts_list: number of external particles
    :param bracket_tokens: Bool for whether we are using composite tokens
    :param func_dict: Dictionary of ab and sb functionals
    :return:
    """

    # Identify the number of external momenta
    if len(npts_list) == 1:
        npt_def = npts_list[0]
    else:
        npt_def = None

    # Call the relevant function based on whether we are using a single token for the square and angle brackets
    if not bracket_tokens:
        return sp_to_mma_single_token(sp_expr, npt_def)
    else:
        if func_dict is None:
            raise AttributeError("Need the function dictionary to evaluate with bracket tokens")
        return sp_to_mma_bracket_token(sp_expr, func_dict, npt_def)


def sp_to_mma_bracket_token(sp_expr, func_dict, npt_def=None):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    Assumes that the relevant variables have been previously initialized
    Assumes that each bracket is a single token
    :param sp_expr: amplitude expression in sympy format
    :param func_dict: Dictionary of ab and sb functionals
    :param npt_def: number of external particles
    :return:
    """
    # List of bracket tokens in the expression
    brackets = list(sp_expr.free_symbols)

    # If we don't have a definite number of external particles
    if npt_def is None:
        momentum_set = set([int(bracket.name[-1]) for bracket in brackets] + [int(bracket.name[-2]) for bracket in brackets])
        n_dep = len(momentum_set)
        npt = max(momentum_set)

        # If we did not get enough momenta show it
        if n_dep < 4:
            logger.info("Got an expression which depends on less than 4 momenta")
    else:
        npt = npt_def

    # Generate the appropriate momenta labels
    args_npt = [sp.Symbol('a{}{}'.format(npt, i)) for i in range(1, npt + 1)]

    # Replace each bracket token by an appropriate sympy string
    dict_replace = {}
    for bracket in brackets:
        name_bk = bracket.name[0:2]
        rule = func_dict[name_bk](args_npt[int(bracket.name[-2]) - 1], args_npt[int(bracket.name[-1]) - 1])
        dict_replace.update({bracket: rule})
    sp_expr = sp_expr.subs(dict_replace)

    # Convert the sympy string to mathematica and convert the bracket operators to the S@M notation
    mma_str = sp.mathematica_code(sp_expr)
    mma_str = mma_str.replace('sb', 'Spbb').replace('ab', 'Spaa')

    return mma_str


def sp_to_mma_single_token(sp_expr, npt_def=None):
    """
    Convert a sympy spinor-helicity expression to a form that can be read by the S@M package
    Assumes that the relevant variables have been previously initialized
    Assumes that each token corresponds to a single word
    :param sp_expr: amplitude expression in sympy format
    :param npt_def: number of external particles
    :return:
    """

    # If we don't have a definite number of external particles
    if npt_def is None:
        func_list = list(sp_expr.atoms(sp.Function))
        momentum_set = set(sum([func.args for func in func_list], ()))
        n_dep = len(momentum_set)
        npt = max([int(momentum.name[-1]) for momentum in list(momentum_set)])

        # If we did not get enough momenta show it
        if n_dep < 4:
            logger.error(str(sp_expr))
            logger.error("Got an expression which depends on less than 4 momenta")
    else:
        npt = npt_def

    # Replace each bracket token by an appropriate sympy string
    args_npt = [sp.Symbol('a{}{}'.format(npt, i)) for i in range(1, npt + 1)]
    replace_dict_var = {sp.Symbol('p{}'.format(i)): args_npt[i - 1] for i in range(1, npt + 1)}
    sp_expr = sp_expr.subs(replace_dict_var)

    # Convert the sympy string to mathematica and convert the bracket operators to the S@M notation
    mma_str = sp.mathematica_code(sp_expr)
    mma_str = mma_str.replace('sb', 'Spbb').replace('ab', 'Spaa')

    return mma_str


def check_numerical_equiv_mma(session, mma_hyp, mma_tgt):
    """
    Check the numerical equivalence between the hypothesis and the target
    :param session: mathematica session
    :param mma_hyp: Amplitude Hypothesis in Mathematica notation
    :param mma_tgt: Target amplitude in Mathematica notation
    :return: return the validity of the hypothesis and the numerical relative difference to the target
    """

    # Call mathematica with the S@M package to check the numerical difference
    res_diff = session.evaluate(wlexpr('Abs[N[(({})-({}))]]'.format(mma_hyp, mma_tgt)))

    # Call mathematica with the S@M package to check the absolute value of the target amplitude
    res_tgt = session.evaluate(wlexpr('Abs[N[{}]]'.format(mma_tgt)))

    # If the target is close to 0 (e.g target is vanishing) then we define the relative difference as the absolute one
    if res_tgt < 10**(-ZERO_ERROR_POW):
        res_rel = res_diff
    else:
        try:
            res_rel = res_diff / res_tgt

        # In case of DirectedInfinity[1]
        except:
            print(res_diff)
            print(res_tgt)
            return False, res_tgt

    try:
        # Check whether the difference is smaller than the given error threshold
        valid = res_rel < 10**(-ZERO_ERROR_POW)
    except:
        return False, res_rel

    # If we have an invalid hypothesis we check whether it is simply off by a sign factor
    if not valid:
        res_add = session.evaluate(wlexpr('Abs[N[(({})+({}))]]'.format(mma_hyp, mma_tgt)))
        if res_add/res_tgt < 10**(-ZERO_ERROR_POW):
            logger.info("We got the wrong overall sign")
            return False, -1

    return valid, res_rel


def mma_to_sp_string_sm(mma_expr):
    """
    Given a Mathematica expression, derived in the S@M package we return a valid Sympy string
    :param mma_expr: Amplitude expression in Mathematica string
    :return:
    """

    # Replace the angle brackets (Can have either momenta as arguments or Sp[momenta])
    sp_str = re.sub(r'Spaa\[Sp\[(\d+)\],\s*Sp\[(\d+)\]\]', r'ab(\1,\2)', mma_expr)
    sp_str = re.sub(r'Spaa\[(\d+),\s*(\d+)\]', r'ab(\1,\2)', sp_str)

    # Replace the square brackets
    sp_str = re.sub(r'Spbb\[Sp\[(\d+)\],\s*Sp\[(\d+)\]\]', r'sb(\1,\2)', sp_str)
    sp_str = re.sub(r'Spbb\[(\d+),\s*(\d+)\]', r'sb(\1,\2)', sp_str)

    # Replace the power signs
    sp_str = sp_str.replace('^', '**')

    return sp_str


def mma_to_sp_string_bk(mma_expr):
    """
    Given a Mathematica expression, with ab[i,j] or sb[i,j] brackets we return a valid Sympy string
    :param mma_expr: Amplitude expression in Mathematica string
    :return:
    """

    # Replace the angle brackets (Can have either momenta as arguments or Sp[momenta])
    sp_str = re.sub(r'ab\[Sp\[(\d+)\],\s*Sp\[(\d+)\]\]', r'ab(\1,\2)', mma_expr)
    sp_str = re.sub(r'ab\[(\d+),\s*(\d+)\]', r'ab(\1,\2)', sp_str)

    # Replace the square brackets
    sp_str = re.sub(r'sb\[Sp\[(\d+)\],\s*Sp\[(\d+)\]\]', r'sb(\1,\2)', sp_str)
    sp_str = re.sub(r'sb\[(\d+),\s*(\d+)\]', r'sb(\1,\2)', sp_str)

    # Replace the power signs
    sp_str = sp_str.replace('^', '**')

    return sp_str


def create_response_frame(hyp_list, envir):
    """
    Given a list of Hypothesis generated in the Streamlit App we return a dataframe with
    the relevant info
    :param hyp_list: List of Amplitude hypothesis with corresponding sympy string and score
    :param envir: Spinor Helicity Environment
    :return:
    """

    # Create the response frame using pandas
    data_in = pd.DataFrame(hyp_list, columns=['Valid Hypothesis', 'Sympy String', 'Score'])

    # Add the latex representation and the mathematica representation
    data_in['Latex String'] = data_in['Sympy String'].apply(latex)
    data_in['Mathematica String'] = data_in['Sympy String'].apply(sp_to_mma, args=(envir.npt_list, envir.bracket_tokens,
                                                                                   envir.func_dict))
    return data_in
