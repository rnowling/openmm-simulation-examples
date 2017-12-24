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

def run_corr(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    backbone = traj.topology.select_atom_indices("minimal")
    traj.superpose(traj, atom_indices=backbone)

    print "computing displacements"
    alpha_carbons = traj.topology.select_atom_indices("alpha")
    traj = traj.atom_slice(alpha_carbons)
    displacement = np.sqrt(np.sum((traj.xyz - np.mean(traj.xyz, axis=0))**2, axis=2))

    if args.disp_matrix_fl:
        np.save(args.disp_matrix_fl,
                displacement)

    print "Computing correlation matrix"
    corr = np.corrcoef(displacement, rowvar=0)

    if args.figures_dir:
        print "Plotting correlation matrix"
        if not os.path.exists(args.figures_dir):
            os.makedirs(args.figures_dir)
        plt.pcolor(corr, vmin=-1.0, vmax=1.0)
        plt.colorbar()
        plt.tight_layout()
        fig_flname = os.path.join(args.figures_dir, "corr.png")
        plt.savefig(fig_flname,
                    DPI=300)


    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figures-dir",
                        type=str,
                        help="Figure output directory")

    parser.add_argument("--disp-matrix-fl",
                        type=str,
                        help="Output displacement matrix")

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

    run_corr(args)
