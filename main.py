import numpy as np
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
        if params.numerical_check:
            session = initialize_numerical_check(params.max_npt, lib_path=params.lib_path)
            evaluator.add_mathematica_session(session)
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        if params.numerical_check:
            end_wolfram_session(session)
        exit()

    # training
    for _ in range(params.max_epoch):
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
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/dumped/',
        'exp_id': 'npt8-infos',
        'save_periodic': 0,
        'tasks': 'spin_hel',

        # environment parameters
        'env_name': 'char_env',
        'max_npt': 8,
        'max_scale': 0.5,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': True,
        'int_base': 10,
        'max_len': 512,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 3,
        'n_dec_layers': 3,
        'n_heads': 8,
        'dropout': 0,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'reload_model': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8-infos/checkpoint.pth',

        # Trainer param
        'export_data': True,
        #'reload_data': 'spin_hel,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8-infos/data.prefix.counts.test,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8-infos/data.prefix.counts.test,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt8-infos/data.prefix.counts.test',
        'reload_data': '',
        'reload_size': '',
        'epoch_size': 10000,
        'max_epoch': 500,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': 'valid_func_simple_acc',
        'reload_checkpoint': '',
        'env_base_seed': -1,
        'batch_size': 1,

        # Evaluation
        'eval_only': False,
        'numerical_check': False,
        'eval_verbose': 2,
        'eval_verbose_print': True,
        'beam_eval': False,
        'beam_size': 1,
        'beam_length_penalty': 1,
        'beam_early_stopping': True,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 1,
        'debug_slurm': False,
        'lib_path': '/Users/aurelien/Documents/Package_lib/Spinors-1.0',

    })

    check_model_params(parameters)
    main(parameters)