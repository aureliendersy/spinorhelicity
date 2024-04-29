from subprocess import call

n_points = [4, 5, 6, 7, 8]
n_ids = [1, 2, 3, 4, 5]
n_par = [1, 2, 3, 4]
for point in n_points:
    for ids in n_ids:
        for par in n_par:
            job_name = 'test_n{}_i{}_r{}'.format(point, ids, par)
            command_dir = 'mkdir -p experiments/data_spin_hel/{}'.format(job_name)
            call(command_dir, shell=True)
            command = 'sbatch --job-name={} data_all.sh {:d} {:d} {}'.format(job_name, point, ids, job_name)
            call(command, shell=True)
import os
from subprocess import call

name_exp_root = 'n4_all'
datadir = 'n4_all'
npoints = '[4]'
beam_sizes = [50]
max_len = 1000

greedy_eval = True

nucleus_sample = False
nucl_ts = [1]
nucl_ps = [0.9]

if greedy_eval:
    name_exp = name_exp_root + '/greedy'
    assert os.path.isdir('final_data/{}'.format(datadir))
    assert os.path.isdir('final_models/{}_{}'.format(name_exp_root, max_len))
    command_dir = 'mkdir -p experiments/eval_simplifier/{}'.format(name_exp)
    call(command_dir, shell=True)
    command = 'sbatch --job-name={} greedy_simplifier.sh {} {} {} {}'.format(name_exp, npoints, datadir,
                                                                                      name_exp, max_len)
    call(command, shell=True)
    exit()

assert nucleus_sample or (len(nucl_ts) == 1 and len(nucl_ps) == 1)
for beam_size in beam_sizes:

    for nucl_t in nucl_ts:
        for nucl_p in nucl_ps:

            if nucleus_sample:
                name_exp = name_exp_root + '/ns_{}_t{}_p{}'.format(beam_size, nucl_t, nucl_p)
            else:
                name_exp = name_exp_root + '/bs_{}'.format(beam_size)
            assert os.path.isdir('final_data/{}'.format(datadir))
            assert os.path.isdir('final_models/{}_{}'.format(name_exp_root, max_len))
            command_dir = 'mkdir -p experiments/eval_simplifier/{}'.format(name_exp)
            call(command_dir, shell=True)
            if nucleus_sample:
                command = 'sbatch --job-name={} nucl_simplifier.sh {} {} {} {} {} {} {}'.format(name_exp, npoints, datadir, name_exp,
                                                                                          beam_size, max_len, nucl_p, nucl_t)
            else:
                command = 'sbatch --job-name={} beam_simplifier.sh {} {} {} {} {}'.format(name_exp, npoints, datadir, name_exp,
                                                                                          beam_size, max_len)
            call(command, shell=True)

