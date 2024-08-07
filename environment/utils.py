# Adapted in part from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Utility functions
"""

import os
import re
import sys
import math
import pickle
import random
import logging
import sympy as sp
import numpy as np
import time
from datetime import timedelta
import subprocess

import errno
import signal
from functools import wraps, partial
from sympy import Function
from add_ons.mathematica_utils import sp_to_mma, check_numerical_equiv_mma, initialize_numerical_check
from add_ons.numerical_evaluations import check_numerical_equiv_local

class LogFormatter:

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


CUDA = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return args
    return [None if x is None else x.cuda() for x in args]


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def convert_momentum_info(infos, max_range):
    """
    Read the identity information vector and convert to the momentum list
    :param infos: the information vector
    :param max_range: Number of external particles
    :return:
    """
    list_info_new = []

    for info in infos:
        info_new = []
        for element in info:
            if not ('ab' in element or 'sb' in element):
                for i in range(1, max_range + 1):
                    element = element.replace('{}'.format(i), 'p{}'.format(i))
            info_new.append(element)
        list_info_new.append(info_new)

    return list_info_new


def check_numerical_equiv_file(prefix_file_path, env, lib_path, infos_in_file=True, check_method=1):
    """
    Safeguard check for verifying that all the examples are properly well-defined
    :param prefix_file_path: Path to training data that contains prefix data
    :param infos_in_file: Whether scrambling identities information is included
    :param env: CharEnv
    :param lib_path: Path to the S@M library
    :param check_method: Check with Mathematica (1) or with local numerical evaluation (2)
    :return:
    """

    if check_method == 1:
        # Start the Mathematica session
        session = initialize_numerical_check(env.max_npt, lib_path=lib_path)
    str_add = "with Mathematica" if  check_method == 1 else "locally"
    print("Reading from {} and checking {}".format(prefix_file_path, str_add))

    counter = 0
    all_correct = True

    with open(prefix_file_path) as infile:
        for line in infile:
            # Convert the prefix expressions to sympy  (account for whether identity info is included)
            if infos_in_file:
                sp2 = env.infix_to_sympy(env.prefix_to_infix(line.split('\t')[1].split('&')[0][:-1].split(' ')))
            else:
                sp2 = env.infix_to_sympy(env.prefix_to_infix(line.split('\t')[1][:-1].split(' ')))

            sp1 = env.infix_to_sympy(env.prefix_to_infix((line.split('|')[-1]).split('\t')[0].split(' ')))

            if check_method == 1:
                # Convert to Mathematica the simple and scrambled amplitudes
                mma2 = sp_to_mma(sp2, env.npt_list, env.func_dict)
                mma1 = sp_to_mma(sp1, env.npt_list,  env.func_dict)
                # Check the numerical equivalence
                matches, res_left = check_numerical_equiv_mma(session, mma1, mma2)
            else:
                # Check the numerical equivalence locally
                matches, res_left = check_numerical_equiv_local(env.special_tokens, sp1, sp2, npt=env.npt_list[0])

            # Output any discrepancy
            if not matches:
                print('Residue is {}'.format(res_left))
                print('Example {} did not match'.format(counter))
                print('Simple expr {}'.format(sp_to_mma(sp2, env.npt_list,  env.func_dict)))
                print('Shuffled expr {}'.format(sp_to_mma(sp1, env.npt_list, env.func_dict)))
                all_correct = False

            counter += 1

            if counter % 100 == 0:
                print("Did {} lines".format(counter))

    if all_correct:
        print('All matches')
    else:
        print('Had some errors in data generation')


def convert_sp_forms(sp_expr, func_dict):
    """
    Given a sympy form using composite tokens, convert it to the regular sympy form
    that uses the ab and sb functionals
    :param sp_expr: Sympy expression of an amplitude
    :param func_dict: Dictionary with the ab, sb functionals
    :return:
    """
    replace_dict = {}
    # If we get the wrong format
    if isinstance(sp_expr, list):
        return None
    # Look at the list of composite tokens and convert them
    for symbol in sp_expr.free_symbols:
        replace_dict.update({symbol: func_dict[symbol.name[0:2]](symbol.name[2], symbol.name[3])})

    return sp_expr.subs(replace_dict)


def revert_sp_form(sp_expr):
    """
    Given a sympy form using ab and sb functionals, revert it to the regular sympy form
    :param sp_expr: Sympy expression of an amplitude
    :return:
    """
    replace_dict = {}
    # If we get the wrong format
    if isinstance(sp_expr, list):
        return None
    # Look at the list of ab and sb functionals and revert them back to composite tokens
    for functional in sp_expr.atoms(sp.Function):
        replace_dict.update({functional: sp.Symbol(str(functional.func) + ''.join(map(str, functional.args)))})

    return sp_expr.subs(replace_dict)


def generate_random_bk(bk_fct, n_points, rng):
    """Provided with the bracket type, generate a bracket with random momenta"""
    pi = rng.randint(1, n_points + 1)
    pj = rng.choice([i for i in range(1, n_points + 1) if i not in [pi]])
    if pi < pj:
        return bk_fct(pi, pj)
    else:
        return bk_fct(pj, pi)


def reorder_expr(hel_expr):
    """
    Reorder the arguments of an expression in canonical form
    Also changes the sign appropriately
    :param hel_expr:
    :return:
    """
    func_list = list(hel_expr.atoms(Function))
    replace_dict = {}
    for fun in func_list:
        # If non-canonical ordering we swap the momenta and multiply by -1
        if fun.args[0] > fun.args[1]:
            new_func = fun.func(fun.args[1], fun.args[0])
            replace_dict.update({fun: new_func*(-1)})

    return_expr = hel_expr.subs(replace_dict)

    return return_expr


def get_n_point(spin_hel_exp):
    """
    Get the n point dependence of an amplitude
    :param spin_hel_exp:
    :return:
    """
    brackets = list(spin_hel_exp.free_symbols)
    return max(set([int(bracket.name[-1]) for bracket in brackets] + [int(bracket.name[-2]) for bracket in brackets]))


def build_scale_factor(scale_list, abfunc, sbfunc, n_points):
    """
    Given a list of factors builds the correct spinor-helicity expression
    Assumes canonical form already
    :param scale_list:
    :param abfunc:
    :param sbfunc:
    :param n_points:
    :return:
    """
    bk_list = [abfunc(i, j) for i in range(1, n_points) for j in range(i+1, n_points+1)]\
            + [sbfunc(i, j) for i in range(1, n_points) for j in range(i+1, n_points+1)]

    ret_expr = 1

    for i, coeff in enumerate(scale_list):
        ret_expr = ret_expr * bk_list[i]**coeff

    return ret_expr


def get_numerator_lg_scaling(sp_numerator, func_dict, npt=5):
    """
    Fast method to get the scaling for only numerator terms
    :param sp_numerator: Sympy expression for the numerator of a spinor-helicity amplitude
    :param func_dict: Dictionary of functionals ab and sb
    :param npt: Number of external particles
    :return:
    """

    # Initialize the return vectors
    scalings = [0] * npt
    mass_dim = 0

    # If we have an addition of terms we take the first one only (they should all have the same scaling)
    if isinstance(sp_numerator, sp.Add):
        term = sp_numerator.args[0]
    else:
        term = sp_numerator

    # For an integer the scaling is 0
    if isinstance(term, sp.Integer):
        return [mass_dim] + scalings

    # If the term is a single bracket then update the scalings vectors appropriately based on its type
    elif isinstance(term, func_dict[0]):
        momentas = term.args
        scalings[int(momentas[0]) - 1] = scalings[int(momentas[0]) - 1] + 1
        scalings[int(momentas[1]) - 1] = scalings[int(momentas[1]) - 1] + 1
        mass_dim += 1
    elif isinstance(term, func_dict[1]):
        momentas = term.args
        scalings[int(momentas[0]) - 1] = scalings[int(momentas[0]) - 1] - 1
        scalings[int(momentas[1]) - 1] = scalings[int(momentas[1]) - 1] - 1
        mass_dim += 1

    # If we have a bracket raised to some power
    elif isinstance(term, sp.Pow):
        func, power = term.args
        mass_dim += power
        momentas = func.args

        if isinstance(func, func_dict[0]):
            sign = 1
        else:
            sign = -1
        scalings[int(momentas[0]) - 1] = scalings[int(momentas[0]) - 1] + sign * power
        scalings[int(momentas[1]) - 1] = scalings[int(momentas[1]) - 1] + sign * power

    # If we have a multiplication of brackets
    elif isinstance(term, sp.Mul):

        # Loop through all brackets and update appropriately
        for arg in term.args:

            # If the brackets are raised to some power
            if isinstance(arg, sp.Pow):
                func, power = arg.args
            # Overall integers don't change the scaling
            elif isinstance(arg, sp.Integer):
                continue
            else:
                power = 1
                func = arg
            mass_dim += power
            momentas = func.args

            if isinstance(func, func_dict[0]):
                sign = 1
            else:
                sign = -1
            scalings[int(momentas[0]) - 1] = scalings[int(momentas[0]) - 1] + sign * power
            scalings[int(momentas[1]) - 1] = scalings[int(momentas[1]) - 1] + sign * power

    return [mass_dim] + scalings


def get_expression_lg_scaling(sp_expression, func_dict, npt=5):
    """
    Fast method to get the scaling for an expression
    :param sp_expression: Sympy expression for a spinor-helicity amplitude
    :param func_dict: Dictionary of functionals ab and sb
    :param npt: Number of external particles
    :return:
    """
    # Split into numerator and denominator and get the scaling of both
    num, denom = sp.fraction(sp_expression)
    num_scalings = np.array(get_numerator_lg_scaling(num, func_dict, npt))
    denom_scalings = np.array(get_numerator_lg_scaling(denom, func_dict, npt))

    return num_scalings - denom_scalings


def get_expression_detail_lg_scaling(sp_expression, func_dict, npt=5):
    """
    Fast method to get the scaling for an expression. Returns the scalings of the numerator and denominator
    independently
    :param sp_expression: Sympy expression for a spinor-helicity amplitude
    :param func_dict: Dictionary of functionals ab and sb
    :param npt: Number of external particles
    :return:
    """
    num, denom = sp.fraction(sp_expression)
    num_scalings = get_numerator_lg_scaling(num, func_dict, npt)
    denom_scalings = get_numerator_lg_scaling(denom, func_dict, npt)

    return num_scalings, denom_scalings
