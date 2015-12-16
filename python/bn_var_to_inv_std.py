import numpy as np
import sys
import os
import os.path as osp
from argparse import ArgumentParser

import caffe


def main(args):
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    eps = 1e-5
    for name, param in net.params.iteritems():
        if name.endswith('_bn'):
            var = param[3].data
            inv_std = 1. / np.sqrt(var + eps)
            param[3].data[...] = inv_std
    net.save(args.output)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This script converts a legacy BN model to new version. "
                    "Specifically, the stored running variance is replace by "
                    "1. / sqrt(var + eps).")
    parser.add_argument('model', help="The deploy prototxt")
    parser.add_argument('weights', help="The caffemodel")
    parser.add_argument('--output', '-o', help="Output caffemodel")
    args = parser.parse_args()
    main(args)
