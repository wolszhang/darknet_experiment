import os
import math
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Barabasi-Albert graph generator')
    parser.add_argument('-n', '--n_nodes', type=int, default=32,
                        help="number of nodes for random graph")
    parser.add_argument('-m', '--m_nodes', type=int, required=True,
                        help="initial number of nodes for random graph")
    parser.add_argument('-o', '--out_txt', type=str, required=True,
                        help="name of output txt file")
    args = parser.parse_args()
    n, m = args.n_nodes, args.m_nodes

    assert 1 <= m < n, "m must be smaller than n."

    deg = np.zeros(n)
    os.makedirs('model', exist_ok=True)
    with open(os.path.join('model', args.out_txt), 'w') as f:
        f.write('Barabasi-Albert\n')
        f.write(str(n) + '\n')

        for i in range(m):
            f.write('%d %d\n' % (m, i))
            deg[m] += 1
            deg[i] += 1

        for i in range(m+1, n):
            edges = np.random.choice(range(n), size=m, replace=False,
                                            p=deg/np.sum(deg))
            for e in edges:
                f.write('%d %d\n' % (e, i))
                deg[e] += 1
                deg[i] += 1
