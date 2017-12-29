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

import numpy as np
import mdtraj as md


import argparse
from itertools import combinations
import os
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hbonds(args):
    residue_pairs = set()
    for r1 in args.residue_group_1:
        for r2 in args.residue_group_2:
            residue_pairs.add(tuple(sorted((r1, r2))))

    print residue_pairs

    print "Reading frames"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "Computing hbonds"
    n_windows = int(np.ceil(traj.n_frames / float(args.window_size)))
    hbond_freq = dict()
    for pair in residue_pairs:
        hbond_freq[pair] = np.zeros(n_windows)

    all_residues = set()
    for r1, r2 in residue_pairs:
        all_residues.add(r1)
        all_residues.add(r2)
    
    selection = " or ".join(["(resid %s)" % r for r in all_residues])
    print selection
    selected_atom_indices = traj.topology.select(selection)
    traj = traj.atom_slice(selected_atom_indices)

    print traj.n_atoms, traj.n_residues
    
    for i in xrange(traj.n_frames):
        hbonds = md.baker_hubbard(traj[i],
                                  periodic=False)

        if i % 1000 == 0:
            print "Frame", (i+1)

        seen_pairs = set()
        for donor_idx, hydrogen_idx, acceptor_idx in hbonds:
            res1 = traj.topology.atom(donor_idx).residue.index
            res2 = traj.topology.atom(acceptor_idx).residue.index
            key = tuple(sorted((res1, res2)))
            if key in residue_pairs and key not in seen_pairs:
                window_idx = i / args.window_size
                hbond_freq[key][window_idx] += 1.0
                seen_pairs.add(key)

    for pair, freq in hbond_freq.iteritems():
        freq /= float(args.window_size)
    
    print "plotting"
    if args.plot_type == "timeseries":
        time = np.arange(1, n_windows + 1) * traj.timestep * args.window_size / 10.
        for i, ((r1, r2), freq) in enumerate(hbond_freq.iteritems()):
            label = "%s -- %s" % (r1 + 1, r2 + 1)
            plt.plot(time, freq, label=label)
            plt.xlabel('Time (ns)', fontsize=16)
            plt.ylabel('H-bond Frequency', fontsize=16)
            plt.ylim([-0.01, 1.01])
    elif args.plot_type == "distribution":
        for i, ((r1, r2), freq) in enumerate(hbond_freq.iteritems()):
            label = "%s -- %s" % (r1 + 1, r2 + 1)
            sns.distplot(freq, kde=False, label=label)
            plt.ylabel('Counts', fontsize=16)
            plt.xlabel('H-bond Frequency', fontsize=16)
    plt.legend()
    plt.savefig(args.figure_fl,
                DPI=300)
    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figure-fl",
                        type=str,
                        required=True,
                        help="Figure output file")

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Input PDB file")

    parser.add_argument("--input-traj",
                        type=str,
                        required=True,
                        help="Input trajectory file")

    parser.add_argument("--residue-group-1",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Residue group 1")

    parser.add_argument("--residue-group-2",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Residue group 2")

    parser.add_argument("--plot-type",
                        type=str,
                        choices=["timeseries",
                                 "distribution"],
                        required=True)

    parser.add_argument("--window-size",
                        type=int,
                        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    plot_hbonds(args)
