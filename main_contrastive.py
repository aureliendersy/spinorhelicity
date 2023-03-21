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
        if params.batch_scalings:
            create_batched_split(env, params, params.prefix_path, 5000)
        else:
            convert_spinor_data(params.prefix_path, ['M', 'S'], env)
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
        'exp_name': 'Test_train_contrastive',
        'dump_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/dumped/',
        'exp_id': 'temp_0.25_',
        'save_periodic': 0,
        'tasks': 'contrastive',

        # environment parameters
        'env_name': 'char_env',
        'npt_list': [5],
        'max_scale': 2,
        'max_terms': 1,
        'max_scrambles': 5,
        'save_info_scr': True,
        'save_info_scaling': True,
        'int_base': 10,
        'numeral_decomp': True,
        'max_len': 2048,
        'canonical_form': True,
        'bracket_tokens': True,
        'generator_id': 2,
        'l_scale': 0.75,
        'numerical_check': False,
        'numerator_only': True,
        'reduced_voc': True,

        # model parameters
        'emb_dim': 512,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'n_heads': 8,
        'dropout': 0,
        'head_layers': 2,
        'n_max_positions': 384,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'reload_model': '',

        # Data param
        'export_data': True,
        #'export_data': False,
        #'prefix_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data.prefix.counts.train',
        'prefix_path': '/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts',
        'mma_path': None,

        # Trainer param
        'reload_data': 'contrastive,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.train,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.valid,/Users/aurelien/PycharmProjects/spinorhelicity/experiments/npt5-infos_a/data_contrastive.prefix.counts.test',
        'reload_size': '',
        'epoch_size': 5000,
        'max_epoch': 500,
        'amp': -1,
        'fp16': False,
        'accumulate_gradients': 1,
        'optimizer': "adam,lr=0.0001",
        'clip_grad_norm': 5,
        'stopping_criterion': '',
        'validation_metrics': '',
        'reload_checkpoint': '',
        'env_base_seed': 1,
        #'env_base_seed': -1,
        'batch_size': 32,
        'temp_contrastive': 0.25,
        'batch_scalings': True,

        # Evaluation
        'eval_only': False,

        # SLURM/GPU param
        'cpu': True,
        'local_rank': -1,
        'master_port': -1,
        'num_workers': 0,
        'debug_slurm': False,

    })

    main(parameters)
