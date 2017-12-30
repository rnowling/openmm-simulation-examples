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

    if args.feature_type == "positions":
        print "aligning frames"
        backbone = traj.topology.select_atom_indices("minimal")
        traj.superpose(traj, atom_indices=backbone)

        print "computing displacements"
        alpha_carbons = traj.topology.select_atom_indices("alpha")
        traj = traj.atom_slice(alpha_carbons)
        features = np.sqrt(np.sum((traj.xyz - np.mean(traj.xyz, axis=0))**2, axis=2))
        
    elif args.feature_type == "dihedrals":
        _, phi_angles = md.compute_phi(traj,
                                   periodic=False)
        _, psi_angles = md.compute_psi(traj,
                                   periodic=False)
        # no dihedral angles for one of the ends
        phi_angles = phi_angles.reshape(traj.n_frames, traj.n_residues - 1, -1)
        psi_angles = psi_angles.reshape(traj.n_frames, traj.n_residues - 1, -1)

        dihedrals = np.concatenate([phi_angles, psi_angles],
                                   axis=2)
        
        features = np.sqrt(np.sum((dihedrals - np.mean(dihedrals, axis=0))**2, axis=2))
        
    elif args.feature_type == "transformed-dihedrals":
        _, phi_angles = md.compute_phi(traj,
                                   periodic=False)
        _, psi_angles = md.compute_psi(traj,
                                   periodic=False)

        phi_sin = np.sin(phi_angles)
        phi_cos = np.cos(phi_angles)
        psi_sin = np.sin(psi_angles)
        psi_cos = np.cos(psi_angles)

        # no dihedral angles for one of the ends
        phi_sin = phi_sin.reshape(traj.n_frames, traj.n_residues - 1, -1)
        psi_sin = psi_sin.reshape(traj.n_frames, traj.n_residues - 1, -1)
        phi_cos = phi_cos.reshape(traj.n_frames, traj.n_residues - 1, -1)
        psi_cos = psi_cos.reshape(traj.n_frames, traj.n_residues - 1, -1)

        dihedrals = np.concatenate([phi_sin,
                                    phi_cos,
                                    psi_sin,
                                    phi_cos],
                                   axis=2)
        
        features = np.sqrt(np.sum((dihedrals - np.mean(dihedrals, axis=0))**2, axis=2))

    print "Computing correlation matrix"
    corr = np.abs(np.corrcoef(features, rowvar=0))

    print "Plotting correlation matrix"
    if args.plot_type == "heatmap":
        plt.pcolor(corr, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.tight_layout()
    elif args.plot_type == "distribution":
        import seaborn as sns
        sns.distplot(np.ravel(corr),
                     kde=False)
        plt.xlabel("Association", fontsize=16)
        plt.ylabel("Occurrences", fontsize=16)
    plt.savefig(args.figure_fl,
                DPI=300)
    
def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--figure-fl",
                        type=str,
                        required=True,
                        help="Figure output file")

    parser.add_argument("--feature-type",
                        type=str,
                        choices=["positions",
                                 "dihedrals",
                                 "transformed-dihedrals"],
                        default="transformed-dihedrals",
                        help="Feature type")

    parser.add_argument("--pdb-file",
                        type=str,
                        required=True,
                        help="Input PDB file")

    parser.add_argument("--input-traj",
                        type=str,
                        required=True,
                        help="Input trajectory file")

    parser.add_argument("--plot-type",
                        type=str,
                        default="heatmap",
                        choices=["heatmap",
                                 "distribution"])

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    run_corr(args)
