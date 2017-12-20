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

def plot_rmsf(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    backbone = traj.topology.select_atom_indices("minimal")
    traj.superpose(traj, atom_indices=backbone)

    print "computing secondary structure"
    sec_structures = md.compute_dssp(traj)

    counter = Counter()
    colors = []
    for r in xrange(traj.n_residues):
        counter.clear()
        for f in xrange(traj.n_frames):
            ss = sec_structures[f, r]
            counter[ss] += 1
        code, cnt = counter.most_common(1)[0]
        if code == "H":
            colors.append("m")
        elif code == "E":
            colors.append("y")
        elif code == "C":
            colors.append("g")
        else:
            raise Exception
            

    print "computing RMSF"
    alpha_carbons = traj.topology.select_atom_indices("alpha")
    avg_xyz = np.mean(traj.xyz[:, alpha_carbons, :], axis=0)
    rmsf = np.sqrt(3*np.mean((traj.xyz[:, alpha_carbons, :] - avg_xyz)**2, axis=(0,2)))

    plt.clf()
    plt.bar(xrange(1, len(rmsf) + 1),
            rmsf,
            width=1.0,
            color = colors)
    plt.xlabel("Residue", fontsize=16)
    plt.ylabel("RMSF (nm)", fontsize=16)
    plt.ylim([0, max(rmsf)])
    fig_flname = os.path.join(args.figures_dir, "rmsf.png")
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
