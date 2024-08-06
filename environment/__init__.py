# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""

"""

from logging import getLogger

from environment.char_env import CharEnv


logger = getLogger()

# We register the Environment
ENVS = {
    'char_env': CharEnv,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)

    # tasks
    tasks = [x for x in params.tasks.split(',') if len(x) > 0]
    assert len(tasks) == len(set(tasks)) > 0
    assert all(task in env.TRAINING_TASKS for task in tasks)
    params.tasks = tasks
    logger.info(f'Training tasks: {", ".join(tasks)}')

    return env
