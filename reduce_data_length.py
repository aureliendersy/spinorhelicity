
"""
Limit the length of our input expressions from a pre-existing dataset
"""

import io
import os
import sys


if __name__ == '__main__':

    assert len(sys.argv) == 3

    limit_size = int(sys.argv[2])

    assert limit_size > 0

    in_path = sys.argv[1]
    out_path = sys.argv[1] + '_{}'.format(limit_size)

    assert not os.path.isfile(out_path)
    assert os.path.isfile(in_path)
    print(f"Reading data from {in_path} ...")

    with io.open(in_path, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")

    print(f"Writing train data to {out_path} ...")

    f_out = io.open(out_path, mode='w', encoding='utf-8')

    for i, line in enumerate(lines):
        if (len(line.split('\t')[0].split('|')[-1].split()) < limit_size
                and len(line.split('\t')[1].split()) < limit_size):
            f_out.write(line)
        if i % 1000000 == 0:
            print(i, end='...', flush=True)

    f_out.close()