"""
File containing the necessary modules for generating the data used for contrastive learning
"""

import os
import sys
import io
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import sympy as sp
import torch
from environment.char_env import EnvDataset, CharEnv
from logging import getLogger
from environment.utils import convert_sp_forms, get_scaling_expr_detail, get_numerator_lg_scaling
from copy import deepcopy


logger = getLogger()


def convert_single_numerator_prefix(prefix_form):
    """
    For a given numerator prefix we assume that we have permutation invariance wrt multiplication
    We give back the same prefix form taking out the mul and INT + tokens
    We also remove power tokens, replacing them with simple repetitions to guarantee permutation invariance
    If we have a token 10 we do not convert since our dictionary will not allow permutation invariance for those numbers
    :param prefix_form:
    :return:
    """

    if '10' in prefix_form:
        return prefix_form, False

    ret_prefix = []
    skip_tok = 0

    for i, token in enumerate(prefix_form):
        if token == 'INT+' or token == 'mul' or token == '1' or i == skip_tok:
            continue
        elif token == 'pow':
            power = int(prefix_form[i+3])
            skip_tok = i + 3
            ret_prefix.extend([prefix_form[i+1]]*(power - 1))
        else:
            ret_prefix.extend([token])

    return ret_prefix, True


def get_scaling_list(prefix_expr, envir):

    scaling_list = []
    rest = prefix_expr

    while len(rest) > 0:
        integ, rest = envir._prefix_to_infix(rest)
        scaling_list.append(int(sp.parse_expr(integ)))

    return scaling_list


def get_scaling_id(scale_lst):

    id_scale = 0

    for i, term in enumerate(scale_lst):
        id_scale += term*10**(2*(len(scale_lst)-1-i))

    return id_scale


def invert_scaling_id(scale_id, n_pt):
    """
    For a given scaling id we can invert it to get back the list of scaling coefficients
    :param scale_id:
    :param n_pt:
    :return:
    """

    scale_temp = scale_id
    scalings = []
    for i in range(n_pt + 1):
        power = 10**(2*(i+1))
        half_power = int(power/2)
        coeff = (scale_temp % half_power) - ((scale_temp % power) // half_power) * half_power
        scale_temp = scale_temp - coeff
        scalings.append(int(coeff/(10**(2*i))))

    scalings.reverse()
    return scalings


def convert_spinor_data(filepath, ids_tokens, env):
    """
    Given a prefix file path that contains equations and their simple form we retain
    only equations that are one identity away. This is defined by the tokens ids given.
    :param filepath:
    :param ids_tokens:
    :param env:
    :return:
    """

    file_size = os.path.getsize(filepath)
    pbar = tqdm(total=file_size, unit="MB")

    outpath = filepath.replace('data', 'data_contrastive')
    print('Writing data in {}'.format(outpath))

    # Open the input file for reading
    with open(filepath, 'r') as f_in:
        # Open the output file for writing
        with open(outpath, 'w') as f_out:

            # Loop through each line of the input file and get the number of identities
            while line := f_in.readline():
                pbar.update(sys.getsizeof(line) - sys.getsizeof('\n'))
                idsinfo = line.split('&')[1]
                numids = np.array([idsinfo.count(token) for token in ids_tokens]).sum()
                numids_forbidden = np.array([idsinfo.count(token) for token in ['Z', 'ID']]).sum()

                prefix_start = line.split('\t')[1].split('&')[0][:-1].split(' ')

                # If the number of identities is 1 then we extract the relevant equation
                # Also verify that the simple form had only 1 term (just to be wary of surprises)
                if numids == 1 and numids_forbidden == 0 and 'add' not in prefix_start:
                    eqprefix = line.split('\t')[0].split('|')[-1].split(' ')
                    eqsp = env.infix_to_sympy(env.prefix_to_infix(eqprefix))
                    numerator, _ = sp.fraction(eqsp)

                    # Add the lg scaling information
                    scale_list = get_numerator_lg_scaling(convert_sp_forms(numerator, env.func_dict),
                                                          list(env.func_dict.values()), npt=env.npt_list[0])
                    scale_id = get_scaling_id(scale_list)

                    assert isinstance(numerator, sp.Add)

                    termsnum = [' '.join(env.sympy_to_prefix(term)) for term in numerator.args]
                    f_out.write(f'{scale_id}|')
                    for i, term in enumerate(termsnum):
                        if i == len(termsnum)-1:
                            f_out.write(f'{term}')
                        else:
                            f_out.write(f'{term}\t')
                    f_out.write(f'\n')
                    f_out.flush()
    pbar.close()


def create_batched_split(env, params, pathin, size_set):

    trn_path = pathin + '.train'
    vld_path = pathin + '.valid'
    tst_path = pathin + '.test'

    print(f"Reading data from {pathin} ...")

    with io.open(pathin, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")

    valid_indices = create_indices_valid_set(env, params, pathin, size_set)
    params.env_base_seed = params.env_base_seed + 1
    test_indices = create_indices_valid_set(env, params, pathin, size_set)

    unique_valid_indices = list(dict.fromkeys([idx for idx in valid_indices if idx not in test_indices]))
    print('Generated {} indices for the validation set'.format(len(unique_valid_indices)))
    unique_test_indices = list(dict.fromkeys([idx for idx in test_indices if idx not in valid_indices]))
    print('Generated {} indices for the test set'.format(len(unique_test_indices)))

    print(f"Writing train data to {trn_path} ...")
    print(f"Writing valid data to {vld_path} ...")
    print(f"Writing test data to {tst_path} ...")
    f_train = io.open(trn_path, mode='w', encoding='utf-8')
    f_valid = io.open(vld_path, mode='w', encoding='utf-8')
    f_test = io.open(tst_path, mode='w', encoding='utf-8')

    valid_lines = []
    test_lines = []

    for i, line in enumerate(lines):
        if i in unique_valid_indices:
            valid_lines.append(line)
        elif i in unique_test_indices:
            test_lines.append(line)
        else:
            f_train.write(line)
        if i % 100000 == 0:
            print(i, end='...', flush=True)

    sorted_valid_index = sorted(unique_valid_indices)
    sorted_test_index = sorted(unique_test_indices)
    ordering_valid = [sorted_valid_index.index(x) for x in unique_valid_indices]
    ordering_test = [sorted_test_index.index(x) for x in unique_test_indices]

    for i in ordering_valid:
        f_valid.write(valid_lines[i])

    for j in ordering_test:
        f_test.write(test_lines[j])

    print('Done')
    f_train.close()
    f_valid.close()
    f_test.close()


def create_indices_valid_set(env, params, pathin, size_set):
    """
    Create the validation (or test sets) based on the batched construction
    :param env:
    :param params:
    :param pathin:
    :param size_set:
    :return:
    """
    envdata = EnvDatasetContrastive(env, 'contrastive', True, None, params, pathin)
    envdata.open_dataset()
    envdata.init_rng()
    index_used = []

    for i in range(size_set):
        if envdata.ref_group_indices is None or envdata.rng.randint(2) == 0:
            batch_index = False
            index = envdata.rng.randint(len(envdata.data))
        else:
            batch_index = True
            index = int(envdata.rng.choice(envdata.ref_group_indices))

        batch_id, _ = envdata.data[index]

        index_used.append(index)

        if envdata.ref_group_indices is None:
            envdata.ref_group_indices = deepcopy(envdata.batch_refs[batch_id])
            envdata.ref_group_indices.remove(index)

        if batch_index:
            envdata.ref_group_indices.remove(index)

        if len(envdata.ref_group_indices) == 0:
            envdata.ref_group_indices = None

    print('Generated indices for a set with {} equations'.format(len(index_used)))

    return index_used


class EnvDatasetContrastive(EnvDataset):
    def __init__(self, env, task, train, rng, params, path):
        super().__init__(env, task, train, rng, params, path)
        self.batch_scalings = params.batch_scalings
        self.batch_refs = None
        self.ref_group_indices = None

    def open_dataset(self):
        self.env = CharEnv(self.params)

        if self.params.export_data and not self.batch_scalings:
            raise ValueError('Not generating data directly for contrastive learning for now')

        # generation, or reloading from file
        if self.path is not None:
            assert os.path.isfile(self.path)
            logger.info(f"Loading data from {self.path} ...")
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not self.train:
                    lines = [line.rstrip().split('|') for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == self.reload_size:
                            break
                        if i % self.n_gpu_per_node == self.local_rank:
                            lines.append(line.rstrip().split('|'))
            if self.batch_scalings:
                self.data = [(scale_ids, xy.split('\t')) for _, scale_ids, xy in lines]
                self.batch_refs = defaultdict(list)
                for i, (k, v) in enumerate(self.data):
                    self.batch_refs[k].append(i)
                self.batch_refs.default_factory = None
            else:
                self.data = [xy.split('\t') for _, xy in lines]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        #nb_examples = sum([[len(element_list)]*len(element_list) for element_list in elements], [])
        nb_examples = [len(element_list) for element_list in elements]
        terms_tensor = sum([[torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in term] for term in elements], [])
        batches = self.env.batch_sequences(terms_tensor)
        return batches, torch.LongTensor(nb_examples)

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        if not hasattr(self, 'env'):
            self.open_dataset()
        self.init_rng()
        if self.path is None:
            raise ValueError('Must have a path to read contrastive data')
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            # Draw from the same LG scaling with 50% chance (if available) otherwise draw another new example
            if self.ref_group_indices is None or self.rng.randint(2) == 0:
                index = self.rng.randint(len(self.data))
                batch_index = False
            else:
                index = int(self.rng.choice(self.ref_group_indices))
                batch_index = True
        if self.batch_scalings and self.train:
            batch_id, samples = self.data[index]
            if self.ref_group_indices is None:
                self.ref_group_indices = deepcopy(self.batch_refs[batch_id])
                self.ref_group_indices.remove(index)

            if batch_index:
                self.ref_group_indices.remove(index)

            if len(self.ref_group_indices) == 0:
                self.ref_group_indices = None
        elif self.batch_scalings and not self.train:
            _, samples = self.data[index]
        else:
            samples = self.data[index]
        split_samples = [sample.split() for sample in samples]
        return split_samples
