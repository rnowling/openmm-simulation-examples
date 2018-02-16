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
from collections import Counter
import os
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

import matplotlib.patches as patches

def plot_rmsf(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    backbone = traj.topology.select_atom_indices("minimal")
    traj.superpose(traj, atom_indices=backbone)

    print "computing RMSF"
    alpha_carbons = traj.topology.select_atom_indices("alpha")
    avg_xyz = np.mean(traj.xyz[:, alpha_carbons, :], axis=0)
    rmsf = np.sqrt(3*np.mean((traj.xyz[:, alpha_carbons, :] - avg_xyz)**2, axis=(0,2)))

    plt.clf()
    plt.plot(xrange(1, traj.n_residues + 1),
             rmsf,
             label="RMSF")

    binding = patches.Rectangle((71, 0),
                                263 - 70,
                                0.1,
                                linewidth=1,
                                edgecolor='y',
                                facecolor='y')
    
    tm1 = patches.Rectangle((264, 0),
                            286 - 264,
                            0.1,
                            linewidth=1,
                            edgecolor='m',
                            facecolor='m')

    tm2 = patches.Rectangle((295, 0),
                            317 - 295,
                            0.1,
                            linewidth=1,
                            edgecolor='m',
                            facecolor='m')

    tm3 = patches.Rectangle((327, 0),
                            349 - 327,
                            0.1,
                            linewidth=1,
                            edgecolor='m',
                            facecolor='m')

    tm4 = patches.Rectangle((520, 0),
                            537 - 520,
                            0.1,
                            linewidth=1,
                            edgecolor='m',
                            facecolor='m')

    ax = plt.gca()
    ax.add_patch(binding)
    ax.add_patch(tm1)
    ax.add_patch(tm2)
    ax.add_patch(tm3)
    ax.add_patch(tm4)
    
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("RMSF (nm)", fontsize=16)
    plt.ylim([0, max(rmsf) * 1.1])
    plt.xlim([-1, traj.n_residues + 2])
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

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    plot_rmsf(args)
