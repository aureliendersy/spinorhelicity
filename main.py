import sys
import numpy as np
import torch
import json
from environment.utils import initialize_exp, AttrDict, convert_to_bracket_file, check_numerical_equiv_file
from environment import build_env
from add_ons.slurm import init_signal_handler, init_distributed_mode
from model import build_modules, check_model_params
import environment
from training.trainer import Trainer
from training.evaluator import Evaluator
from add_ons.mathematica_utils import initialize_numerical_check, end_wolfram_session, initialize_solver_session

np.seterr(all='raise')


def main(params):
    #convert_to_bracket_file("/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8_exp2/data.prefix.counts.test")
    #exit()

    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()
    environment.utils.CUDA = not params.cpu

    env = build_env(params)

    #check_numerical_equiv_file('/Users/aurelien/PycharmProjects/spinorhelicity/experiments/dumped/Test_data_spin_hel/test/data.prefix', env, params.lib_path)
    #exit()

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
        'exp_name': 'Test_eval_spin_hel',
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/dumped/',
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
        'save_info_scr': False,
        'save_info_scaling': False,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 1000,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
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
        #'reload_model': '',
        #'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/temp_data/n5_i1_3_1000/checkpoint.pth',
        #'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/checkpoint.pth',
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/final_notebooks/Trained_Models/Run2/Simplifier_Models/5pt/n5_i1_3_l1000_10Mex.pth',

        # Trainer param
        #'export_data': True,
        'export_data': False,
        #'reload_data': '',
        #'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/temp_data/data.prefix.counts.test_1000,/Users/aurelien/PycharmProjects/spinorhelicity/temp_data/data.prefix.counts.test_1000,/Users/aurelien/PycharmProjects/spinorhelicity/temp_data/data.prefix.counts.test_1000',
        #'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5/data.prefix.counts.valid',
        'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/final_notebooks/Data/Run2/n5_i1_3/test_n5_i1_3_1000,/Users/aurelien/PycharmProjects/spinorhelicity/final_notebooks/Data/Run2/n5_i1_3/test_n5_i1_3_1000,/Users/aurelien/PycharmProjects/spinorhelicity/final_notebooks/Data/Run2/n5_i1_3/test_n5_i1_3_1000',
        'reload_size': '',
        'epoch_size': 1000,
        'max_epoch': 10,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 2,

        # Evaluation
        'eval_only': True,
        #'eval_only': False,
        'test_file': True,
        'valid_file': False,
        'numerical_check': 2,
        'eval_verbose': 0,
        'eval_verbose_print': False,
        'beam_eval': False,
        'beam_size': 1,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,
        'nucleus_sampling': False,
        'nucleus_p': 0.95,
        'temperature': 2,
        'scaling_eval': True,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '/Users/aurelien/Documents/Package_lib/Spinors-1.0',
        #'mma_path': 'NotRequired',
        'mma_path': None,
    })

    check_model_params(parameters)
    main(parameters)
