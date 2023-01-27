# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

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
from add_ons.mathematica_utils import sp_to_mma, check_numerical_equiv, initialize_numerical_check


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


def convert_to_momentum(sp_expr, momentum_list):
    """
    Convert the momentum in a spinor helicity amplitude from a integer to a momentum label
    :param sp_expr:
    :param momentum_list:
    :return:
    """
    func_list = list(sp_expr.atoms(sp.Function))

    replace_dict = {i: momentum_list[i - 1] for i in range(1, len(momentum_list) + 1)}
    replace_dict_func = {func_list[i]: func_list[i].subs(replace_dict) for i in range(len(func_list))}

    sp_expr = sp_expr.subs(replace_dict_func)

    return sp_expr


def convert_momentum_info(infos, max_range, skip_bk):
    """
    Read the identity information vector and convert to the momentum list
    :param infos:
    :param max_range:
    :param skip_bk
    :return:
    """
    list_info_new = []

    for info in infos:
        info_new = []
        for element in info:
            if not(skip_bk and ('ab' in element or 'sb' in element)):
                for i in range(1, max_range + 1):
                    element = element.replace('{}'.format(i), 'p{}'.format(i))
            info_new.append(element)
        list_info_new.append(info_new)

    return list_info_new


def convert_to_bracket_tokens(prefix_expr):
    """
    Take as input a prefix expression and convert the brackets into a single token
    :param prefix_expr:
    :return:
    """

    return_expr = []
    pass_word = 0

    for i, word in enumerate(prefix_expr):
        if word in ['ab', 'sb']:
            return_expr.append(word + prefix_expr[i+1].replace('p', '') + prefix_expr[i+2].replace('p', ''))
            pass_word += 2
        elif pass_word == 0:
            return_expr.append(word)
        else:
            pass_word -= 1

    return return_expr


def convert_to_bracket_file(prefix_file_path):
    """
    Read a file with the prefix data and convert it to a new alphabet
    :param prefix_file_path:
    :return:
    """

    print("Reading from {}".format(prefix_file_path))

    out_path = prefix_file_path + '_new_alphabet'
    new_file = open(out_path, "w")

    counter = 0

    with open(prefix_file_path) as infile:
        for line in infile:
            prefix2_str = ' '.join(convert_to_bracket_tokens(line.split('\t')[1][:-1].split(' ')))
            prefix1_str = ' '.join(convert_to_bracket_tokens(line.split('\t')[0].split(' ')))
            new_file.write(f'{prefix1_str}\t{prefix2_str}\n')
            counter += 1

            if counter % 1000 == 0:
                print("Did {} lines".format(counter))

    new_file.close()


def check_numerical_equiv_file(prefix_file_path, env, lib_path):
    """
    Safeguard check for verifying that all the examples are properly well defined
    :param prefix_file_path:
    :param env
    :param lib_path
    :return:
    """

    session = initialize_numerical_check(env.max_npt, lib_path=lib_path)

    print("Reading from {}".format(prefix_file_path))

    counter = 0

    with open(prefix_file_path) as infile:
        for line in infile:
            sp2 = env.infix_to_sympy(env.prefix_to_infix(line.split('\t')[1][:-1].split(' ')))
            mma2 = sp_to_mma(sp2, env.bracket_tokens, env.func_dict)
            sp1 = env.infix_to_sympy(env.prefix_to_infix((line.split('|')[-1]).split('\t')[0].split(' ')))
            mma1 = sp_to_mma(sp1, env.bracket_tokens, env.func_dict)

            matches, res_left = check_numerical_equiv(session, mma1, mma2)

            if not matches:
                print('Residue is {}'.format(res_left))
                print('Example {} did not match'.format(counter))
                print('Simple expr {}'.format(mma2))
                print('Shuffled expr {}'.format(mma1))

            counter += 1

            if counter % 1 == 0:
                print("Did {} lines".format(counter))


def convert_sp_forms(sp_expr, func_dict):
    """
    Given a sympy form using composite tokens, convert it to the regular sympy form
    that uses the ab and sb functionals
    :param sp_expr:
    :param func_dict:
    :return:
    """

    replace_dict = {}

    if isinstance(sp_expr, list):
        return None

    for symbol in sp_expr.free_symbols:
        replace_dict.update({symbol: func_dict[symbol.name[0:2]](symbol.name[2], symbol.name[3])})

    return sp_expr.subs(replace_dict)


def generate_random_bk(bk_fct, n_points, rng, canonical=False):
    """Provided with the bracket type, generate a bracket with random momenta"""
    pi = rng.randint(1, n_points if canonical else n_points+1)
    if canonical:
        pj = rng.choice([i for i in range(pi+1, n_points + 1)])
    else:
        pj = rng.choice([i for i in range(1, n_points + 1) if i not in [pi]])
    return bk_fct(pi, pj)


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
        if fun.args[0] > fun.args[1]:
            new_func = fun.func(fun.args[1], fun.args[0])
            replace_dict.update({fun: new_func*(-1)})

    return_expr = hel_expr.subs(replace_dict)

    return return_expr


def get_scaling_expr(spin_hel_expr, func_list):
    """
    Given a spinor helicity expression we figure out the number of angle brackets
    and square brackets in the numerator and denominator
    :param spin_hel_expr:
    :param func_list:
    :return:
    """
    if isinstance(spin_hel_expr, sp.Add):
        expr_f = spin_hel_expr.args[0]
    else:
        expr_f = spin_hel_expr

    # Separate out the numerator and the denominator
    num, denom = sp.fraction(expr_f)

    # To deal with brackets to higher power we replace them by some placeholder value
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    map_dict = {func_list[0]: x1, func_list[1]: x2}
    repl_rule = {bk: map_dict[bk.func] for bk in expr_f.atoms(sp.Function)}
    num_subs = num.subs(repl_rule)
    denom_subs = denom.subs(repl_rule)

    return [sp.total_degree(num_subs, x1), sp.total_degree(num_subs, x2), sp.total_degree(denom_subs, x1),
            sp.total_degree(denom_subs, x2)]


def get_scaling_expr_detail(spin_hel_expr, func_list, n_point):
    """
        Given a spinor helicity expression we figure out the little group scaling
        for each momentum along with the mass dimension. Return it for the
        numerator and denominator as two vectors, starting with the mass dimension
        :param spin_hel_expr:
        :param func_list:
        :param n_point:
        :return:
        """

    if isinstance(spin_hel_expr, sp.Add):
        expr_f = spin_hel_expr.args[0]
    else:
        expr_f = spin_hel_expr

    # Separate out the numerator and the denominator
    num, denom = sp.fraction(expr_f)

    return get_lg_ms(num, func_list, n_point), get_lg_ms(denom, func_list, n_point)


def get_lg_ms(in_expr, func_list, n_point):
    """
    Get the mass dimension along with the little group scaling of an expression
    :param in_expr:
    :param func_list:
    :param n_point:
    :return:
    """

    # To deal with brackets to higher power we replace them by some placeholder value
    xvars = [sp.Symbol('x{}'.format(i + 1)) for i in range(n_point)]
    map_dict_ab = {func_list[0](i + 1, j + 1): xvars[i] * xvars[j] for i in range(n_point) for j in range(n_point)}
    map_dict_sb = {func_list[1](i + 1, j + 1): 1 / (xvars[i] * xvars[j]) for i in range(n_point) for j in
                   range(n_point)}

    ms = sp.Symbol('ms')
    dict_mass_scale = {bk: ms for bk in in_expr.atoms(sp.Function)}

    expr_subs = in_expr.subs(map_dict_ab).subs(map_dict_sb)
    expr_subs_ms = in_expr.subs(dict_mass_scale)

    lg_degrees = [sp.total_degree(sp.fraction(expr_subs)[0], xvar) - sp.total_degree(sp.fraction(expr_subs)[1], xvar) for xvar in xvars]
    expr_ms = sp.total_degree(expr_subs_ms, ms)

    return [expr_ms] + lg_degrees


def get_helicity_expr(spin_hel_exp, func_list):
    """
    Get the total helicity by looking at its little group scaling
    :param spin_hel_exp:
    :param func_list:
    :return:
    """

    x1 = sp.Symbol('x1')
    map_dict = {func_list[0]: x1, func_list[1]: 1/x1}
    repl_rule = {bk: map_dict[bk.func]*bk for bk in spin_hel_exp.atoms(sp.Function)}
    repl_rule_num = {bk: np.random.random_sample() for bk in spin_hel_exp.atoms(sp.Function)}
    scale_expr = spin_hel_exp.subs(repl_rule)
    res_num = (sp.log((scale_expr / spin_hel_exp).subs(repl_rule_num)) / sp.log(x1)).subs(
        {x1: np.random.random_sample()})
    try:
        int_res = round(res_num)

    except:
        return 'Undefined'

    if abs(int_res - res_num) < 10**(-8):
        return int_res
    else:
        return 'Undefined'


def get_n_point(spin_hel_exp):
    """
    Get the n point dependence of an amplitude
    :param spin_hel_exp:
    :return:
    """
    brackets = list(spin_hel_exp.free_symbols)
    return max(set([int(bracket.name[-1]) for bracket in brackets] + [int(bracket.name[-2]) for bracket in brackets]))


def random_scale_factor(scale_list, abfunc, sbfunc, n_points, rng, canonical=False):
    """
    Given the scaling list we generate a random appropriate scaling factor
    :param scale_list:
    :param abfunc:
    :param sbfunc:
    :param n_points
    :param rng
    :param canonical
    :return:
    """

    ret_expr = 1
    for i in range(scale_list[0]):
        ret_expr *= generate_random_bk(abfunc, n_points, rng, canonical=canonical)
    for j in range(scale_list[1]):
        ret_expr *= generate_random_bk(sbfunc, n_points, rng, canonical=canonical)
    for k in range(scale_list[2]):
        ret_expr *= 1/generate_random_bk(abfunc, n_points, rng, canonical=canonical)
    for l in range(scale_list[3]):
        ret_expr *= 1/generate_random_bk(sbfunc, n_points, rng, canonical=canonical)

    return ret_expr


def build_scale_factor(scale_list, abfunc, sbfunc, n_points):
    """
        Given a list of factors build the correct spin helicity expression
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


def add_scaling_lg(lg_scale_vector, bk_add, num):
    """
    Add the correct little group scaling to the existing vector
    :param lg_scale_vector:
    :param bk_add:
    :param num:
    :return:
    """

    sign = 1 if bk_add.func.__name__ == 'sb' else -1
    sign = sign if num else -sign

    for arg in bk_add.args:
        lg_scale_vector[arg - 1] += sign

    return lg_scale_vector
