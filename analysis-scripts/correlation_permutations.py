"""
Copyright 2017 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import numpy as np

def calculate_correlations(args):
    displacements = np.load(args.disp_matrix_fl)

    print "Computing permutations"
    if not os.path.exists(args.corr_dir):
        os.makedirs(args.corr_dir)

    n_residues = displacements.shape[1]
    for i in xrange(args.n_permutations):
        print "Computing permutation", i
        for res_id in xrange(n_residues):
            np.random.shuffle(displacements[:, i])
        
        print "Computing correlations"
        permuted_corr = np.corrcoef(displacements, rowvar=0)
        print

        flname = os.path.join(args.corr_dir,
                              "corr_matrix_%s.npy" % i)
        np.save(flname,
                permuted_corr)


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--disp-matrix-fl",
                        type=str,
                        required=True,
                        help="Output displacement matrix")

    parser.add_argument("--n-permutations",
                        type=int,
                        required=True,
                        help="Number of permutations")

    parser.add_argument("--corr-dir",
                        type=str,
                        required=True,
                        help="Output correlations")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    calculate_correlations(args)
