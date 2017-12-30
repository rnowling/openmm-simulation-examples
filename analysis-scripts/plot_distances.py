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
            residue_pairs.add((r1, r2))

    print residue_pairs

    print "Reading frames"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "Computing contacts"
    distances, pairs = md.compute_contacts(traj,
                                           contacts=list(residue_pairs),
                                           scheme="closest",
                                           periodic=False)

    print "plotting"
    if args.plot_type == "timeseries":
        time = np.arange(1, distances.shape[0] + 1) * traj.timestep / 10.
        for i, (r1, r2) in enumerate(pairs):
            res1 = traj.topology.residue(r1)
            res2 = traj.topology.residue(r2)
            label = "%s -- %s" % (res1, res2)
            plt.plot(time, distances[:, i], label=label)
            plt.xlabel('Time (ns)', fontsize=16)
            plt.ylabel('Residue Distance (nm)', fontsize=16)
            plt.ylim([0.0, np.nanmax(distances) + 0.2])
    elif args.plot_type == "distribution":
        for i, (r1, r2) in enumerate(pairs):
            res1 = traj.topology.residue(r1)
            res2 = traj.topology.residue(r2)
            label = "%s -- %s" % (res1, res2)
            sns.distplot(distances[:, i], kde=False, label=label)
            plt.ylabel('Counts', fontsize=16)
            plt.xlabel('Residue Distance (nm)', fontsize=16)
    plt.legend()
    plt.savefig(args.figures_fl,
                DPI=300)
    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figures-fl",
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

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    plot_hbonds(args)
