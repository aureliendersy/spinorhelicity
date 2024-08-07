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
from environment.utils import convert_sp_forms, get_numerator_lg_scaling
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

    # If we have an integer greater than 10 we skip
    if '10' in prefix_form:
        return prefix_form, False

    ret_prefix = []
    skip_tok = 0

    for i, token in enumerate(prefix_form):
        # Don't output integer or multiplication tokens
        if token == 'INT+' or token == 'mul' or token == '1' or i == skip_tok:
            continue
        # For power tokens we repeat the token the appropriate number of times
        elif token == 'pow':
            power = int(prefix_form[i+3])
            skip_tok = i + 3
            ret_prefix.extend([prefix_form[i+1]]*(power - 1))
        else:
            ret_prefix.extend([token])

    return ret_prefix, True


def get_scaling_id(scale_lst):
    """
    For a given list of little group scaling coefficients we create a unique id
    :param scale_lst:
    :return:
    """
    id_scale = 0

    # For each scaling we create a unique identifier by multiplying each scaling by a given power
    # Power indices are spaced by 2 to account for scalings that go up to 100
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


def convert_spinor_data(filepath, ids_tokens, env, check_ids=False):
    """
    Given a prefix file path that contains equations and their simple form we retain
    only equations that are one identity away. This is defined by the tokens ids given.
    :param filepath:
    :param ids_tokens:
    :param env:
    :param check_ids:
    :return:
    """

    # Create a progress bar based on the file size
    file_size = os.path.getsize(filepath)
    pbar = tqdm(total=file_size, unit="MB")

    # Define the path where we output the data
    outpath = filepath.replace('data', 'data_contrastive')
    print('Writing data in {}'.format(outpath))

    # Open the input file for reading
    with open(filepath, 'r') as f_in:
        # Open the output file for writing
        with open(outpath, 'w') as f_out:

            # Loop through each line of the input file and get the number of identities
            while line := f_in.readline():
                pbar.update(sys.getsizeof(line) - sys.getsizeof('\n'))

                # If we want to check whether the equation is generated using a single identity
                if check_ids:
                    idsinfo = line.split('&')[1]
                    numids = np.array([idsinfo.count(token) for token in ids_tokens]).sum()
                    numids_forbidden = np.array([idsinfo.count(token) for token in ['Z', 'ID']]).sum()

                prefix_start = line.split('\t')[1].split('&')[0][:-1].split()

                # If the number of identities is 1 then we extract the relevant equation
                # Also verify that the simple form had only 1 term (just to be wary of surprises)
                valid_id_line = numids == 1 and numids_forbidden == 0 if check_ids else True

                if valid_id_line and 'add' not in prefix_start:

                    # Convert the input to a sympy form and extract the numerator and denominator
                    eqprefix = line.split('\t')[0].split('|')[-1].split(' ')
                    eqsp = env.infix_to_sympy(env.prefix_to_infix(eqprefix))
                    numerator, denominator = sp.fraction(eqsp)

                    # Do the same for the expected output
                    eqsptgt = env.infix_to_sympy(env.prefix_to_infix(prefix_start))
                    numerator_tgt, denominator_tgt = sp.fraction(eqsptgt)

                    # If the target is not 0 then we put it and the input on some common denominator
                    # Choose that denominator to be minimal
                    if numerator_tgt != 0:
                        relative_denominator = denominator/denominator_tgt
                        rel_denom1, rel_denom2 = sp.fraction(relative_denominator)

                        # If output denominator is a subset of input denominator then scaling is easy
                        if rel_denom2 == 1:
                            numerator_tgt = numerator_tgt * rel_denom1
                        else:
                            numerator_tgt = numerator_tgt * rel_denom1
                            numerator = numerator * rel_denom2

                    combined_numerator = numerator - numerator_tgt

                    # Add the lg scaling information by converting it to a unique ID
                    scale_list = get_numerator_lg_scaling(convert_sp_forms(combined_numerator, env.func_dict),
                                                          list(env.func_dict.values()), npt=env.npt_list[0])
                    scale_id = get_scaling_id(scale_list)

                    # If we have multiple numerator terms in the input then we aggregate them
                    if isinstance(combined_numerator, sp.Add):
                        termsnum = [' '.join(env.sympy_to_prefix(term)) for term in combined_numerator.args]
                    else:
                        termsnum = [' '.join(env.sympy_to_prefix(combined_numerator))]

                    # Write all the numerators separated by tab and preceded by the scaling ID
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
    """
    Given a path to a file we create a train, valid and test set
    The validation and test sets are not part of the train set and created by selecting examples
    that have similar little group scaling
    :param env:
    :param params:
    :param pathin:
    :param size_set:
    :return:
    """

    # Define the paths to output the data
    trn_path = pathin + '.train'
    vld_path = pathin + '.valid'
    tst_path = pathin + '.test'

    print(f"Reading data from {pathin} ...")

    # Read the data of numerator terms in prefix notation
    with io.open(pathin, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")

    # Obtain relevant indices for the validation and test set (that mix expressions with the same LG scaling)
    valid_indices = create_indices_valid_set(env, params, pathin, size_set)
    params.env_base_seed = params.env_base_seed + 1
    test_indices = create_indices_valid_set(env, params, pathin, size_set)

    # Assure that the indices generated are not found in both sets
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

    # Write the appropriate indices in the appropriate files
    # For the test and valid files we write later - after sorting
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

    # Write the validation and test set
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

    # Build the contrastive environment
    envdata = EnvDatasetContrastive(env, 'contrastive', True, None, params, pathin)
    envdata.open_dataset()
    envdata.init_rng()
    index_used = []

    for i in range(size_set):
        # If we have a group then select a random index half the time
        # If we don't have a group then always select it randomly
        if envdata.ref_group_indices is None or envdata.rng.randint(2) == 0:
            batch_index = False
            index = envdata.rng.randint(len(envdata.data))
        # Select the index from the appropriate group otherwise
        else:
            batch_index = True
            index = int(envdata.rng.choice(envdata.ref_group_indices))

        # Recover the id of the batch and retain the example index used
        batch_id, _ = envdata.data[index]
        index_used.append(index)

        # Update the group indices that we have considered so far and remove the one considered
        if envdata.ref_group_indices is None:
            envdata.ref_group_indices = deepcopy(envdata.batch_refs[batch_id])
            envdata.ref_group_indices.remove(index)

        if batch_index:
            envdata.ref_group_indices.remove(index)

        # If we have no more relevant examples in the group indices then we mark the group as done
        # Also do that on a random basis to make sure that the entire LG scaling group does not
        # populate the current split
        if len(envdata.ref_group_indices) == 0 or envdata.rng.randint(20) == 0:
            envdata.ref_group_indices = None

    print('Generated indices for a set with {} equations'.format(len(index_used)))

    return index_used


def convert_file_to_permutation_inv(inputpath):
    """
    Given an input file we open it and read it line by line.
    For each line we extract the prefix expressions (separated by tabs) and convert them into
    a form that is permutation invariant. Then we write back the obtained result.
    For now not used in practice but consider for future work
    :param inputpath:
    :return:
    """

    # Read the input data
    print(f"Reading data from {inputpath} ...")
    with io.open(inputpath, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")

    # Define the output path
    outpath = inputpath + '.perm'
    print(f"Writing data to {outpath} ...")
    f_out = io.open(outpath, mode='w', encoding='utf-8')

    valid_tot = 0
    for i, line in enumerate(lines):
        if i % 100000 == 0:
            print(i, end='...', flush=True)
        parts_line = line.replace('\n', '').split('|')

        # Retain the individual numerator prefixes
        prefix_parts = parts_line[-1].split('\t')
        convert_prefix = []

        skip_line = False
        for prefix_part in prefix_parts:
            # Convert the prefix to a permutation invariant form
            convert_pre, success = convert_single_numerator_prefix(prefix_part.split(' '))
            if success:
                convert_prefix.append(' '.join(convert_pre))
            # If we cannot convert (e.g integer greater than 10 is present then we skip this example  line)
            else:
                skip_line = True
        if skip_line:
            continue
        else:
            valid_tot += 1

        parts_line[-1] = '\t'.join(convert_prefix)
        out_line = '|'.join(parts_line)
        f_out.write(out_line + '\n')

    print('Wrote {} lines'.format(valid_tot))
    print('Done')
    f_out.close()


class EnvDatasetContrastive(EnvDataset):
    def __init__(self, env, task, train, rng, params, path):
        super().__init__(env, task, train, rng, params, path)
        self.batch_scalings = params.batch_scalings
        self.batch_refs = None
        self.ref_group_indices = None

    def open_dataset(self):
        """
        Open the dataset, either by generating it or by loading it from file
        :return:
        """
        # Create the base enviorment
        self.env = CharEnv(self.params)

        if self.params.export_data and not self.batch_scalings:
            raise ValueError('Not generating data directly for contrastive learning for now')

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

            # If we want to create batches by IDS we extract the scaling ID
            if self.batch_scalings:
                self.data = [(scale_ids, xy.split('\t')) for _, scale_ids, xy in lines]
                self.batch_refs = defaultdict(list)

                # Create a mapping dictionary associating IDs to example number
                for i, (k, v) in enumerate(self.data):
                    self.batch_refs[k].append(i)
                self.batch_refs.default_factory = None
            else:
                self.data = [xy.split('\t') for _, xy in lines]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

    def collate_fn(self, elements):
        """
        Collate a batch (list) of samples into a batch of tensors.
        """
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

            # Take a new scaling group if the reference is empty or 5% of the time randomly
            if len(self.ref_group_indices) == 0 or self.rng.randint(20) == 0:
                self.ref_group_indices = None

        # In evaluation mode we can just read through the batch in the natural order
        elif self.batch_scalings and not self.train:
            _, samples = self.data[index]
        else:
            samples = self.data[index]

        # Return the different numerator terms considered
        split_samples = [sample.split() for sample in samples]
        return split_samples
