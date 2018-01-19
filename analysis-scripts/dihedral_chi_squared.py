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

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from scipy import stats

from sklearn.externals import joblib

def run_test(args):
    if len(args.boundaries) != 4:
        raise Exception, "Need four boundary points"

    n_bins = 10
    bins = np.linspace(-np.pi, np.pi, num=n_bins + 1)
    print bins
    
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    _, phi_angles = md.compute_phi(traj,
                                   periodic=False)
    _, psi_angles = md.compute_psi(traj,
                                   periodic=False)

    if args.boundaries:
        phi_1 = phi_angles[args.boundaries[0]:args.boundaries[1], :]
        psi_1 = psi_angles[args.boundaries[0]:args.boundaries[1], :]
        phi_2 = phi_angles[args.boundaries[2]:args.boundaries[3], :]
        psi_2 = psi_angles[args.boundaries[2]:args.boundaries[3], :]
    elif args.state_labels and args.states:
        labels = joblib.load(args.state_labels)
        state_1 = []
        state_2 = []
        for i, label in enumerate(labels):
            if label == args.states[0]:
                state_1.append(i)
            if label == args.states[1]:
                state_2.append(i)
                
        phi_1 = phi_angles[state_1, :]
        psi_1 = psi_angles[state_1, :]
        phi_2 = phi_angles[state_2, :]
        psi_2 = psi_angles[state_2, :]
    else:
        raise Exception, "need to specify boundaries or state_labels and states"

    log_pvalues = []

    for resid in xrange(traj.n_residues - 1):
        dist_1, _, _ = np.histogram2d(phi_1[:, resid],
                                      psi_1[:, resid],
                                      bins = [bins, bins])

        dist_2, _, _ = np.histogram2d(phi_2[:, resid],
                                      psi_2[:, resid],
                                      bins = [bins, bins])

        # fudge factor to ensure that no bins are empty
        dist_1 += 1

        freq_1 = (dist_1 / np.sum(dist_1)).flatten()

        freq_2 = (dist_2 / np.sum(dist_2)).flatten()

        G = 0
        used_bins = 0
        for i in xrange(freq_1.shape[0]):
            # skip over empty bins
            if freq_2[i] > 0.0:
                used_bins += 1
                G += freq_2[i] * np.log(freq_2[i] / freq_1[i])
        G *= 2 * dist_2.size

        p = stats.chi2.sf(G, used_bins)

        print resid, G, p

        upper_bound = 100.0
        if p == 0.0:
            log_p = upper_bound
        else:
            log_p = -np.log10(p)
            if log_p > upper_bound:
                log_p = upper_bound
        log_pvalues.append(log_p)

    plt.plot(np.arange(1, len(log_pvalues) + 1),
              log_pvalues)
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("p-value (-log)", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(args.figure_fl,
                DPI=300)
    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figure-fl",
                        type=str,
                        required=True)

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Input PDB file")

    parser.add_argument("--input-traj",
                        type=str,
                        required=True,
                        help="Input trajectory file")

    parser.add_argument("--boundaries",
                        type=int,
                        nargs=4,
                        help="Frame boundaries")

    parser.add_argument("--state-labels",
                        type=str,
                        help="Mapping of frames to states")

    parser.add_argument("--states",
                        type=int,
                        nargs=2,
                        help="States to use in comparison")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    run_test(args)
