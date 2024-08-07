# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Environment used to deal with the tokenization of spinor-helicity amplitudes
Calls on the generator from the helicity environment to generate original sympy expresssions and tokenizes them
"""

from logging import getLogger
import os
import io
import sys
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.calculus.util import AccumBounds
from environment.bracket_env import ab, sb
from environment.utils import timeout, TimeoutError, convert_momentum_info
from environment.helicity_generator import generate_random_amplitude
from environment.spin_helicity_env import SpinHelExpr
from add_ons.mathematica_utils import initialize_solver_session

CLEAR_SYMPY_CACHE_FREQ = 10000


SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '&']
SPECIAL_WORDS2 = ['<s>', '</s>', '<pad>']

logger = getLogger()


class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class CharEnv(object):

    TRAINING_TASKS = {'spin_hel', 'contrastive'}

    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',

        # Brackets
        ab: 'ab',
        sb: 'sb',
    }

    OPERATORS = {
        # Elementary functions
        'add': 2,
        'mul': 2,
        'pow': 2,

        # Allow for div to represent rationals (even if we don't train on them directly)
        'div': 2,

        # Brackets
        'ab': 2,
        'sb': 2,
    }

    def __init__(self, params):

        # Mathematica Parameters
        self.session = None
        self.mma_path = params.mma_path
        self.numerical_check = params.numerical_check

        # Number of external parameters
        self.npt_list = params.npt_list
        self.max_npt = max(self.npt_list)

        # Representation of integers
        self.int_base = params.int_base
        self.numeral_decomp = params.numeral_decomp

        # Parameters for the amplitude generation
        self.max_len = params.max_len
        self.max_scale = params.max_scale
        self.max_terms = params.max_terms
        self.max_scrambles = params.max_scrambles
        self.min_scrambles = params.min_scrambles
        self.l_scale = params.l_scale
        self.all_momenta = params.all_momenta

        # Parameters for recording the representation of amplitudes and their properties
        self.save_info_scr = params.save_info_scr
        self.save_info_scaling = params.save_info_scaling
        self.numerator_only = params.numerator_only
        self.reduced_voc = params.reduced_voc

        assert self.max_npt >= 4
        assert abs(self.int_base) >= 2

        # Don't need the ab and sb operators if we are using combined brackets
        self.operators = sorted([el for el in list(self.OPERATORS.keys()) if el != 'ab' and el != 'sb'])

        # Possible constants and variables. The variables used are to denote the momenta.
        # (Leave for possible future work - not required in the minimal vocabulary)
        # For the constants we use letters to represent the type of identity used
        self.constants = ['S', 'M', 'M+', 'M-', 'ID+', 'ID-', 'Z+', 'Z-']
        self.func_dict = {'ab': ab, 'sb': sb}
        self.variables = OrderedDict({
            'p{}'.format(i): sp.Symbol('p{}'.format(i)) for i in range(1, self.max_npt + 1)})

        # When using a reduced vocabulary we generate only the bracket with canonical ordering of  momenta
        if self.reduced_voc:
            self.special_tokens = list(filter(None, ['ab{}{}'.format(i, j) if i != j else None
                                                     for i in range(1, self.max_npt + 1)
                                                     for j in range(i+1, self.max_npt + 1)])) \
                                  + list(filter(None, ['sb{}{}'.format(i, j) if i != j else None
                                                       for i in range(1, self.max_npt + 1)
                                                       for j in range(i+1, self.max_npt + 1)]))
        else:
            self.special_tokens = list(filter(None, ['ab{}{}'.format(i, j) if i != j else None
                                                     for i in range(1, self.max_npt+1)
                                                     for j in range(1, self.max_npt+1)]))\
                                  + list(filter(None, ['sb{}{}'.format(i, j) if i != j else None
                                                       for i in range(1, self.max_npt+1)
                                                       for j in range(1, self.max_npt+1)]))
        # Define the symbols and tokens required to represent integers
        self.symbols = ['INT+', 'INT-']
        self.elements = [str(i) for i in range(abs(self.int_base))]
        if self.numeral_decomp:
            self.elements = self.elements + ['10']
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(self.variables.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # Complete word vocabulary - in practice we use the reduced version for training the models
        if self.reduced_voc and not (self.save_info_scr or self.save_info_scaling):
            self.words = SPECIAL_WORDS2 + self.operators + self.symbols + self.elements + self.special_tokens
        else:
            self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) +\
                         self.operators + self.symbols + self.elements + self.special_tokens

        # Define the mapping between tokens and their respective IDs
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # Number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

    def add_mathematica_session(self, session):
        self.session = session

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        Use a numeral decomposition scheme when relevant
        """
        base = self.int_base
        res = []
        max_digit = abs(base)

        neg = val < 0
        val = -val if neg else val

        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        res.append('INT-' if neg else 'INT+')

        # If we use the decomposition scheme we must split the integer into the base 10 decomposition
        if self.numeral_decomp:
            if len(res) == 2:
                return res[::-1]
            else:
                return self.numeral_decomp_builder(res[0], res[-1], res[1:-1], 0)
        else:
            return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        val = 0

        # Check that the list starts with an integer marker
        if not (base >= 2 and lst[0] in ['INT+', 'INT-']):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0

        # Parse the list and break when we don't have an integer marker
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break

            # Update the integer value with the number encountered
            val = val * base + int(x)
            i += 1

        # Add the minus sign if required
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def numeral_decomp_builder(self, int_in, int_sign, rem_int, order):
        """
        Splits a given integer (along with its overall sign) into a base 10 decomposition
        :param int_in: Input digit
        :param int_sign: Overall sign of the expression
        :param rem_int: Remaining digits to be considered
        :param order: Order in base 10 of int_in
        :return:
        """

        # If we get a zero we move to the next decimal
        if int(int_in) == 0:
            return self.numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
        # If we have no more digits to consider don't return anything extra
        if len(rem_int) == 0:
            return_val = []
        # If we still have digits we append the return value
        else:
            return_val = self.numeral_decomp_builder(rem_int[0], int_sign, rem_int[1:], order + 1)
        # If we have a number smaller than 10
        if order == 0:
            return ['add', int_sign, int_in] + return_val

        add_term = ['add'] if len(rem_int) > 0 else []
        coeff_term = ['mul', int_sign, int_in] if (int(int_in) != 1 or int_sign != 'INT+') else []
        # If we have a number between 10 and 100
        if order == 1:
            return add_term + coeff_term + ['INT+', '10'] + return_val
        # If we have a number greater than 100
        else:
            return add_term + coeff_term + ['pow', 'INT+', '10'] + self.write_int(order) + return_val

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'sb':
            return f'sb({args[0]}, {args[1]})'
        elif token == 'ab':
            return f'ab({args[0]}, {args[1]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in infix mode (returns human readable string)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.variables or t in self.constants or t in self.special_tokens:
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0 and not ((self.save_info_scr and r[0] == '&') or (self.save_info_scaling and r[0] == '&')):
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def infix_to_sympy(self, infix):
        """
        Convert an infix expression to SymPy.
        """
        # Parse the infix string with sympy directly
        expr = parse_expr(infix, evaluate=True, local_dict=self.func_dict)
        if expr.has(AccumBounds):
            logger.error('Expression {} failed. Was {} originally'.format(expr, infix))
            raise ValueErrorExpression
        return expr

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)
        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # parse children
        parse_list = []

        if not op in ['ab', 'sb']:
            for i in range(n_args):
                if i == 0 or (i < n_args - 1):
                    parse_list.append(op)
                parse_list += self.sympy_to_prefix(expr.args[i])

        else:
            combined_tok = op + str(expr.args[0]) + str(expr.args[1])
            assert combined_tok in self.special_tokens
            parse_list += [combined_tok]

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        # For declared sympy symbols or bracket tokens
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        # Parse integer numbers
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        # In the special case where we have an overall rational number for instance
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # Unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def scr_info_to_prefix(self, info_s):
        """
        Convert the information about the identities applied into a parsed prefix expression
        :param info_s:
        :return:
        """
        prefix_ret = []
        for info_vec in info_s:
            for info_w in info_vec:
                # If the information is an identity token marker
                if info_w in self.constants:
                    prefix_ret.append(info_w)
                # If the information is using a momentum token
                elif info_w in self.variables:
                    prefix_ret.extend(self.sympy_to_prefix(sp.parse_expr(info_w)))
                # If the information is another token (e.g a bracket token)
                else:
                    prefix_ret.extend(self.sympy_to_prefix(sp.parse_expr(info_w, local_dict=self.func_dict)))

        return prefix_ret

    def scr_prefix_to_infix(self, infos_prefix):
        """
        Convert the prefix information vector to an infix form
        :param infos_prefix:
        :return:
        """
        out_in = ''
        for word in infos_prefix:
            if word in self.constants:
                if out_in != '' and out_in[-1].isdigit():
                    out_in += ')'
                out_in += '/' + word + ':'
            elif word in self.symbols:
                pass
            elif word in self.operators:
                out_in += word
            elif word in self.variables:
                if out_in[-1].isdigit():
                    out_in += ','
                    next_w = ')'
                else:
                    out_in += '('
                    next_w = ''
                out_in += word
                out_in += next_w
            elif word in self.special_tokens:
                out_in += word
        return out_in

    def scale_info_to_prefix(self, info_scale):
        """
        Convert the information about the scaling of an expression into a parsed prefix expression
        :param info_scale:
        :return:
        """

        # List of coefficients are already converted to sympy integers
        prefix_ret = []
        for info_vec in info_scale:
            prefix_ret.extend(self.sympy_to_prefix(sp.sympify(info_vec)))
        return prefix_ret

    def scale_prefix_to_infix(self, infos_prefix):
        """
        Convert the prefix information vector to an infix form for the scaling
        :param infos_prefix:
        :return:
        """

        p_list = []
        r = infos_prefix

        while len(r) > 0:
            p, r = self._prefix_to_infix(r)
            p_list.append(p)

            if len(p_list) > self.max_npt + 1:
                raise ValueError('Cannot parse the dimension information')

        out_in = ': Scaling : MD: ' + p_list[0] + '/ LG: '
        for i, p_val in enumerate(p_list[1:]):
            out_in = out_in + 'p{}: {},'.format(i+1, p_val)
        return out_in

    @timeout(1000)
    def gen_hel_ampl(self, rng):
        """
        Generate a spinor helicity amplitude and an equivalent scrambled version of it
        """

        try:
            # Generate a simple spinor helicity expression
            # Also retain some information about the amplitude (number of particles and little group scaling)
            simple_expr, n_pt_gen, scale_inf = generate_random_amplitude(self.npt_list, rng,
                                                                         max_terms_scale=self.max_scale,
                                                                         max_components=self.max_terms,
                                                                         l_scale=self.l_scale,
                                                                         info_scaling=self.save_info_scaling,
                                                                         session=self.session,
                                                                         all_momenta=self.all_momenta)
            if simple_expr is None:
                return None

            # Convert the amplitude to the SpinorHelicity Environment and scramble it
            simple_expr_env = SpinHelExpr(str(simple_expr), n_pt_gen)
            info_s = simple_expr_env.random_scramble(rng, max_scrambles=self.max_scrambles, out_info=self.save_info_scr,
                                                     session=self.session, numerator_only=self.numerator_only,
                                                     min_scrambles=self.min_scrambles)

            # Put the resulting amplitude into a fraction form with a single common denominator
            simple_expr_env.cancel()

            # If the scrambled expression drops a momentum then we discard it from the training set
            # We do this if we could have an ambiguity when generating amplitude with different number of particles
            if len(self.npt_list) > 1\
                    and any([i not in np.array([list(f.args)
                                                for f in simple_expr_env.sp_expr.atoms(sp.Function)]).flatten()
                             for i in range(1, simple_expr_env.n_point + 1)]):
                return None

            # Save the identity information if required
            if self.save_info_scr:
                info_s = convert_momentum_info(info_s, self.max_npt)
                prefix_info = self.scr_info_to_prefix(info_s)
            else:
                prefix_info = None

            # Save the scaling information if required
            if self.save_info_scaling:
                prefix_info_scale = self.scale_info_to_prefix(scale_inf)
            else:
                prefix_info_scale = None

            # Convert the simple and scrambled amplitudes to prefix notation
            shuffled_expr = simple_expr_env.sp_expr
            simple_prefix = self.sympy_to_prefix(simple_expr)
            shuffled_prefix = self.sympy_to_prefix(shuffled_expr)

            # Skip too long sequences
            if max(len(simple_prefix), len(shuffled_prefix)) > self.max_len:
                logger.info("Rejected Equation as was too long for {} identities"
                            .format(len([info for info in info_s if 'ID' not in ''.join(info)
                                         and 'Z' not in ''.join(info)])))
                return None

            # Skip equations that do not simplify
            # We consider (ab and ab^2) to be of equal complexity here)
            num_bk_simple = ''.join(simple_prefix).count('sb') + ''.join(simple_prefix).count('ab')
            num_bk_shuffle = ''.join(shuffled_prefix).count('sb') + ''.join(shuffled_prefix).count('ab')
            if num_bk_shuffle <= num_bk_simple:
                return None

        # If we spend too long to generate we skip
        except TimeoutError:
            raise
        except (ValueError, AttributeError, UnknownSymPyOperator, ValueErrorExpression) as e:
            logger.error("Got an exception in line {0} for expression \"{1}\". Arguments:{2!r}.".format(sys.exc_info()[-1].tb_lineno, simple_expr, e.args))
            return None
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, simple_expr, e.args))
            return None

        # If required we append the identity and scaling information to the prefix output
        out_prefix = simple_prefix
        if self.save_info_scr:
            out_prefix = out_prefix + ['&'] + prefix_info
        if self.save_info_scaling:
            out_prefix = out_prefix + ['&'] + prefix_info_scale

        # Define input / output
        return shuffled_prefix, out_prefix

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--max_npt", type=int, default=8,
                            help="Maximum number of external momenta")
        parser.add_argument("--max_terms", type=int, default=1,
                            help="Maximum number of distinct terms")
        parser.add_argument("--max_scale", type=int, default=1,
                            help="Maximum scaling of the length of simple expression compared to # of external momenta")
        parser.add_argument("--max_scrambles", type=int, default=5,
                            help="Maximum number of scrambles applied to an expression")
        parser.add_argument("--int_base", type=int, default=10,
                            help="Integer representation base")

    def create_train_iterator(self, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            rng=None,
            params=params,
            path=(None if data_path is None else data_path[task][0])
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_train_iterator_contrastive(self, dataset, task, params, data_path):

        logger.info(f"Creating train iterator for {task} and environment {self.__class__.__name__}...")

        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator_contrastive(self, dataset, data_type, task, params):

        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for {task} and environment {self.__class__.__name__} ...")

        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator(self, data_type, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            rng=np.random.RandomState(0),
            params=params,
            path=(None if data_path is None else data_path[task][1 if data_type == 'valid' else 2])
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )


class EnvDataset(Dataset):

    def __init__(self, env, task, train, rng, params, path):
        super(EnvDataset).__init__()
        self.params = params
        self.rng = rng
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.n_gpu_per_node = params.n_gpu_per_node
        self.local_rank = params.local_rank
        self.reload_size = params.reload_size
        assert (train is True) == (rng is None)
        assert task in CharEnv.TRAINING_TASKS

        # Batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        # Dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        else:
            assert os.path.isfile(self.path)
            logger.info(f"Preparing to load data from {self.path} ...")
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                lines = [line.rstrip().split('|') for line in f]
            self.size = 5000 if path is None else len(lines)

    def open_dataset(self):
        self.env = CharEnv(self.params)
        # Data generation - Need Mathematica session for the zero identity
        if self.params.export_data and self.params.mma_path != 'NotRequired':
            session = initialize_solver_session(kernel_path=self.params.mma_path)
            self.env.add_mathematica_session(session)
        else:
            self.env.session = 'NotRequired'
        # Generation, or reloading from file
        if self.path is not None:
            assert os.path.isfile(self.path)
            logger.info(f"Loading data from {self.path} ...")
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                # Either reload the entire file, or the first N lines (for the training set)
                if not self.train:
                    lines = [line.rstrip().split('|') for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == self.reload_size:
                            break
                        if i % self.n_gpu_per_node == self.local_rank:
                            lines.append(line.rstrip().split('|'))
            self.data = [xy.split('\t') for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in x]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        if worker_info is None:
            return 0
        else:
            return worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        if not hasattr(self, 'env'):
            self.open_dataset()
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """

        x, y = None, None

        while True:
            try:
                if self.task == 'spin_hel':
                    xy = self.env.gen_hel_ampl(self.rng)
                else:
                    raise Exception(f'Unknown data type: {self.task}')
                if xy is None:
                    continue
                x, y = xy
                break
            except TimeoutError:
                continue
            except Exception as e:
                logger.error("An unknown exception of type {0} occurred for worker {4} in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, 'F', e.args, self.get_worker_id()))
                continue

        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()
        return x, y
