"""
Modules used for the training of the contrastive algorithm
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from training.trainer import Trainer
from environment.contrastive_data import EnvDatasetContrastive
from environment.utils import to_cuda, TimeoutError
from logging import getLogger

logger = getLogger()


def contrastive_loss(encoded, batch_ids, temp):
    """
    Compute the contrastive loss for a set of positive and negative examples. In each group we
    have a number of examples given by batch ids.
    :param encoded: a tensor containing the encoded representations of the examples
    :param batch_ids: a list containing the number of examples in each group (batch)
    :param temp: the temperature parameter used for scaling the similarity scores
    :return:
    """
    # Get the device on which the tensor is located
    device = encoded.device

    # Compute the cosine similarity matrix between all pairs of examples
    cossim = nn.CosineSimilarity(dim=-1)
    similarity_mat = cossim(encoded.unsqueeze(0), encoded.unsqueeze(1))

    # Compute the exponential similarity matrix and the masks for the numerator and denominator
    # In the num we normalize such that each group of positive examples has similar weighting (a pair has weight 1)
    exp_mat = torch.exp(similarity_mat/temp)
    masks_nums = torch.block_diag(*[torch.ones(ids, ids, device=device).fill_diagonal_(0)/(ids-1) for ids in batch_ids])
    masks_denom = (1 - torch.block_diag(*[torch.ones(ids, ids, device=device).fill_diagonal_(0)
                                          for ids in batch_ids])).fill_diagonal_(0)
    numerator = (exp_mat * masks_nums).sum(dim=1)
    denominator = (exp_mat * masks_denom).sum(dim=1)

    # Compute the final contrastive loss and sum
    batch_size = batch_ids.sum()
    loss = -torch.log(numerator/denominator).sum()/batch_size

    return loss


def evaluation_losses(encoded, batch_ids):
    """
    Get the alignement and uniformity losses
    :param encoded:
    :param batch_ids:
    :return:
    """
    # Get the device on which the tensor is located
    device = encoded.device

    # Compute the cosine similarity matrix between all pairs of examples
    cossim = nn.CosineSimilarity(dim=-1)
    similarity_mat = cossim(encoded.unsqueeze(0), encoded.unsqueeze(1))

    # Compute the final contrastive loss and sum
    batch_size = batch_ids.sum()

    # Compute the exponential similarity matrix and the masks for the numerator and denominator
    # In the num we normalize such that we look at the average alignment within each group
    masks_nums = torch.block_diag(
        *[torch.ones(ids, ids, device=device).fill_diagonal_(0) / (ids-1) for ids in batch_ids])
    masks_denom = (1 - torch.block_diag(*[torch.ones(ids, ids, device=device).fill_diagonal_(0)
                                          for ids in batch_ids])).fill_diagonal_(0)

    # Alignment is ||x-y||^2 with x and y normalized (alpha=2)
    alignment_loss = (2 * (1 - similarity_mat) * masks_nums).sum(dim=1).sum() / batch_size

    # Uniform is exp of minus the norm squared (t=2)
    exp_mat = torch.exp(-4*(1-similarity_mat))
    normalization = torch.tensor(sum([[batch_size - ids] * ids for ids in batch_ids], []))
    denominator = ((exp_mat * masks_denom).sum(dim=1) / normalization).sum() / batch_size
    uniform_loss = torch.log(denominator)

    return alignment_loss, uniform_loss


class ContrastiveEvaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        ContrastiveEvaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations, calculating the alignement and uniformity loss
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                for task in self.params.tasks:
                    self.calculate_losses(data_type, task, scores)

        return scores

    def calculate_losses(self, data_type, task, scores):
        """
        Calculate contrastive losses, alignement loss and uniform losses
        """
        params = self.params
        encoder_c = self.modules['encoder_c']
        encoder_c.eval()

        # stats
        c_losses = 0
        align_losses = 0
        uniform_losses = 0
        batches = 0

        # iterator
        dataset = EnvDatasetContrastive(self.env, params.tasks[0], train=False, rng=np.random.RandomState(0),
                                        params=params,
                                        path=(None if self.trainer.data_path is None
                                              else self.trainer.data_path[params.tasks[0]][1 if data_type == 'valid'
                                        else 2]))

        iterator = self.env.create_test_iterator_contrastive(dataset, data_type, task, params=params)

        for (x_batch, len_batch), ids_batch in iterator:

            # cuda
            x_batch, len_batch, ids_batch = to_cuda(x_batch, len_batch, ids_batch)

            # forward / loss
            encoded = encoder_c('fwd', x=x_batch, lengths=len_batch, causal=False)
            c_loss = contrastive_loss(encoded, ids_batch, params.temp_contrastive)
            align_loss, uniform_loss = evaluation_losses(encoded, ids_batch)

            c_losses += c_loss.item()
            align_losses += align_loss.item()
            uniform_losses += uniform_loss.item()
            batches += 1

        logger.info(f"Contrastive loss is:  {c_losses/batches} .")
        logger.info(f"Alignment loss is:  {align_losses/batches} .")
        logger.info(f"Uniform loss is:  {uniform_losses/batches} .")

        scores[f'{data_type}_{task}_c_loss'] = c_losses / batches
        scores[f'{data_type}_{task}_a_loss'] = align_losses / batches
        scores[f'{data_type}_{task}_u_loss'] = uniform_losses / batches


class ContrastiveTrainer(Trainer):

    def __init__(self, modules, env, params):
        super().__init__(modules, env, params)

        # Overwrite the data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)

            # For now support only 1 task
            assert len(params.tasks) == 1
            self.dataset = EnvDatasetContrastive(self.env, params.tasks[0], train=True, rng=None, params=params,
                                                 path=(None if self.data_path is None
                                                       else self.data_path[params.tasks[0]][0]))
            self.dataloader = {
                task: iter(self.env.create_train_iterator_contrastive(self.dataset, task, params, self.data_path))
                for task in params.tasks
            }

    def training_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder_c = self.modules['encoder_c']
        encoder_c.train()

        # batch
        (x_batch, len_batch), ids_batch = self.get_batch(task)

        # cuda
        x_batch, len_batch, ids_batch = to_cuda(x_batch, len_batch, ids_batch)

        # forward / loss
        encoded = encoder_c('fwd', x=x_batch, lengths=len_batch, causal=False)
        loss = contrastive_loss(encoded, ids_batch, params.temp_contrastive)

        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats['processed_e'] += len_batch.size(0)
        self.stats['processed_w'] += len_batch.sum().item()
