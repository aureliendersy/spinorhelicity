"""
Main file for generating/training/evaluating the simplifier model
"""

import sys
import numpy as np
import torch
import json
from environment.utils import initialize_exp, AttrDict
from environment import build_env
from add_ons.slurm import init_signal_handler, init_distributed_mode
from model import build_modules, check_model_params
import environment
from training.trainer import Trainer
from training.evaluator import Evaluator
from add_ons.mathematica_utils import initialize_numerical_check, end_wolfram_session

np.seterr(all='raise')


def main(params):

    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()
    environment.utils.CUDA = not params.cpu

    env = build_env(params)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # evaluation
    if params.eval_only:
        # Set the seed for the nucleus sampling
        torch.random.manual_seed(42)
        if params.numerical_check == 1:
            session = initialize_numerical_check(env.max_npt, lib_path=params.lib_path)
            evaluator.add_mathematica_session(session)
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        if params.numerical_check == 1:
            end_wolfram_session(session)
        exit()

    # training
    for _ in range(params.max_epoch - trainer.epoch):
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                # If we just want to generate data
                if params.export_data:
                    trainer.export_data(task)
                # If we want to train
                else:
                    trainer.enc_dec_step(task)
                    if trainer.n_equations % 1000 == 0:
                        percent_done = round(trainer.n_equations / trainer.epoch_size * 100)
                        logger.info("Did {} % of epoch {}".format(percent_done, trainer.epoch))
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        if not params.export_data:
            # end of epoch
            trainer.save_best_model(scores)
            trainer.save_periodic()
            trainer.end_epoch(scores)
        else:
            trainer.epoch += 1


if __name__ == '__main__':

    parameters = AttrDict({

        # Name
        'exp_name': 'Test_data_spin_hel',
        'dump_path': 'path', # Define your path
        'exp_id': '5pt_test',
        'save_periodic': 0,
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 3,
        'max_scrambles': 3,
        'min_scrambles': 1,
        'save_info_scr': True,
        'save_info_scaling': True,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 1000,
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': True,
        'all_momenta': False,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'n_max_positions': 2560,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'reload_model': '', # Define the path to your model to reload it

        # Trainer param
        'export_data': True, # Set to False if not generating data
        'reload_data': '', # When training on evaluating set  data paths
        'reload_size': '',
        'epoch_size': 1000,
        'max_epoch': 10,
        'amp': -1,  # Change if using apex
        'fp16': False, # Change if using mixed precision
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 2,

        # Evaluation
        'eval_only': False, # Change if evaluating network
        'test_file': True,
        'valid_file': False,
        'numerical_check': 2, # Change if evaluating with Mathematica and not lcoally
        'eval_verbose': 0,
        'eval_verbose_print': False,
        'beam_eval': False,
        'beam_size': 1,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': False,
        'nucleus_p': 0.95,
        'temperature': 1.5,
        'scaling_eval': False,

        # SLURM/GPU param
        'cpu': True, # Change if you have a GPU
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '', # Path to Spinors-1.0 library
        'mma_path': None, # Change None if you need to specify your path to Mathematica (not needed when training)
    })

    check_model_params(parameters)
    main(parameters)
