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

def plot_rmsf(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    backbone = traj.topology.select_atom_indices("minimal")
    traj.superpose(traj, atom_indices=backbone)

    print "computing distances"
    alpha_carbons = traj.topology.select_atom_indices("alpha")
    traj = traj.atom_slice(alpha_carbons)
    
    dist = np.sqrt(np.sum((traj.xyz - np.mean(traj.xyz, axis=0))**2, axis=2))

    print "Computing correlation matrix"
    corr = np.corrcoef(dist)

    print "Plotting correlation matrix"
    plt.pcolor(corr)
    plt.colorbar()
    plt.tight_layout()
    fig_flname = os.path.join(args.figures_dir, "corr.png")
    plt.savefig(fig_flname,
                DPI=300)


    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figures-dir",
                        type=str,
                        required=True,
                        help="Figure output directory")

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
