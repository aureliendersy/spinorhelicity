"""
File containing the necessary modules for generating the data used for contrastive learning
"""

import os
import sys
import io
from tqdm import tqdm
import numpy as np
import sympy as sp
import torch
from environment.char_env import EnvDataset, CharEnv
from logging import getLogger

logger = getLogger()


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

                # If the number of identities is 1 then we extract the relevant equation
                if numids == 1:
                    eqprefix = line.split('\t')[0].split('|')[-1].split(' ')
                    eqsp = env.infix_to_sympy(env.prefix_to_infix(eqprefix))
                    numerator, _ = sp.fraction(eqsp)

                    assert isinstance(numerator, sp.Add)

                    termsnum = [' '.join(env.sympy_to_prefix(term)) for term in numerator.args]
                    for i, term in enumerate(termsnum):
                        if i == len(termsnum)-1:
                            f_out.write(f'{term}')
                        else:
                            f_out.write(f'{term}\t')
                    f_out.write(f'\n')
                    f_out.flush()
    pbar.close()


class EnvDatasetContrastive(EnvDataset):
    def __init__(self, env, task, train, rng, params, path):
        super().__init__(env, task, train, rng, params, path)

    def open_dataset(self):
        self.env = CharEnv(self.params)

        if self.params.export_data:
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
            index = self.rng.randint(len(self.data))
        samples = self.data[index]
        split_samples = [sample.split() for sample in samples]
        return split_samples
