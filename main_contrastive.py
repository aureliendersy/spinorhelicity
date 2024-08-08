"""
Main file for generating/training the embedding model
"""
import numpy as np
import json
from environment.utils import AttrDict
from environment.contrastive_data import convert_spinor_data, create_batched_split
from add_ons.slurm import init_signal_handler, init_distributed_mode
import environment
from environment.utils import initialize_exp
from environment import build_env
from model.contrastive_learner import build_modules_contrastive
from training.contrastive_trainer import ContrastiveTrainer, ContrastiveEvaluator


def main(params):

    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()

    environment.utils.CUDA = not params.cpu
    env = build_env(params)

    if params.export_data:
        pass # Uncomment if you need to generate data from a prefix file
        convert_spinor_data(params.prefix_path, ['M', 'S'], env)
        exit()

    # To generate the train/valid/test split from a file with numerators in prefix form
    if params.batch_scalings and params.export_data:
        create_batched_split(env, params, params.prefix_path, 10000)
        exit()
    else:
        modules = build_modules_contrastive(env, params)
        trainer = ContrastiveTrainer(modules, env, params)
        evaluator = ContrastiveEvaluator(trainer)

        # training
        for _ in range(params.max_epoch):
            logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

            trainer.n_equations = 0

            while trainer.n_equations < trainer.epoch_size:

                # training steps
                for task_id in np.random.permutation(len(params.tasks)):
                    task = params.tasks[task_id]
                    # If we want to train
                    trainer.training_step(task)
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
        'exp_name': 'n5_i1',
        'dump_path': 'path', # Define your own path
        'exp_id': 'train',
        'save_periodic': 0,
        'tasks': 'contrastive',

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
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': True,
        'all_momenta': False,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'n_heads': 8,
        'dropout': 0,
        'head_layers': 2,
        'n_max_positions': 256,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'norm_ffn': 'layernorm',
        'reload_model': '', # Define the model if you want to reload it

        # Data param
        'export_data': True, # Uncomment if training
        'prefix_path': 'prefix_path', # Prefix file required for generating the training data
        'mma_path': None,

        # Trainer param
        'reload_data': False, # Replace with path to data when training
        'reload_size': '',
        'epoch_size': 50,
        'max_epoch': 500,
        'amp': -1, # Change if using apex
        'fp16': False, # Change if using mixed precision
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': 1,
        'batch_size': 32,
        'temp_contrastive': 0.25,
        'batch_scalings': True,

        # Evaluation
        'eval_only': False,
        'numerical_check': 0,

        # SLURM/GPU param
        'cpu': True, # Change if using GPU
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 0,
        'debug_slurm': False,

    })

    main(parameters)
