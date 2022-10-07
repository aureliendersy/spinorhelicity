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


def convert_momentum_info(infos, max_range):
    """
    Read the identity information vector and convert to the momentum list
    :param infos:
    :param max_range:
    :return:
    """
    list_info_new = []

    for info in infos:
        info_new = []
        for element in info:
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
