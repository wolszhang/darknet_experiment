import math
import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Erdos-Renyi graph generator')
    parser.add_argument('-n', '--nodes', type=int, default=32,
                        help="number of nodes for random graph")
    parser.add_argument('-p', '--prob', type=float, default=0.4,
                        help="probablity of node connection for ER")
    parser.add_argument('-o', '--out', type=str, required=True,
                        help="name of output txt")
    args = parser.parse_args()
    nodes, prob = args.nodes, args.prob

    if np.log(nodes) > prob * nodes:
        print("Warning: p must be at least #nodes/log(#nodes) otherwise graph may be disconnected.")

    rand = np.random.uniform(0.0, 1.0, size=(nodes, nodes))

    os.makedirs('model', exist_ok=True)
    with open(os.path.join('model', args.out), 'w') as f:
        f.write('Erdos-Renyi\n')
        f.write(str(nodes) + '\n')

        for i in range(nodes):
            for j in range(i+1, nodes):
                if rand[i][j] < prob:
                    f.write('%d %d\n' % (i, j))
