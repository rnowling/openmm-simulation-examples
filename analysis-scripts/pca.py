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
from itertools import combinations
import os
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

SVD_KEY = "svd"
PROJECTION_KEY = "projected-coordinates"

def compute_pca(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "aligning frames"
    traj.superpose(traj)

    reshaped = traj.xyz.reshape(traj.n_frames,
                                traj.n_atoms * 3)

    print "Fitting SVD"
    svd = TruncatedSVD(n_components = args.n_components)
    projected = svd.fit_transform(reshaped)

    print "Writing model"
    model = { SVD_KEY : svd,
              PROJECTION_KEY : projected }
    
    joblib.dump(model, args.model_file)

def compute_distance_pca(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    print "computing distances"
    alpha_carbons = traj.topology.select_atom_indices("alpha")

    print len(alpha_carbons)

    atom_pairs = list(combinations(alpha_carbons,
                                   2))
    pairwise_distances = md.geometry.compute_distances(traj,
                                                       atom_pairs)
    print pairwise_distances.shape

    print "Fitting SVD"
    svd = TruncatedSVD(n_components = args.n_components)
    projected = svd.fit_transform(pairwise_distances)

    print "Writing model"
    model = { SVD_KEY : svd,
              PROJECTION_KEY : projected }
    
    joblib.dump(model, args.model_file)

def explained_variance_analysis(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    model = joblib.load(args.model_file)
    svd = model[SVD_KEY]

    plt.clf()
    plt.grid(True)
    plt.plot(svd.explained_variance_ratio_, "m.-")
    plt.xlabel("Principal Component", fontsize=16)
    plt.ylabel("Explained Variance Ratio", fontsize=16)
    plt.ylim([0., 1.])
    fig_flname = os.path.join(args.figures_dir, "pca_explained_variance_ratios.png")
    plt.savefig(fig_flname,
                DPI=300)

def pairwise(iterable):
    iterable = iter(iterable)
    try:
        while True:
            a = next(iterable)
            b = next(iterable)
            yield a, b
    except StopIteration:
        pass
    
def plot_projections(args):
    if len(args.pairs) % 2 != 0:
        print "Error: PCs must be provided in pairs of 2"
        sys.exit(1)

    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    model = joblib.load(args.model_file)
    projected = model[PROJECTION_KEY]

    for p1, p2 in pairwise(args.pairs):
        H, xedges, yedges = np.histogram2d(projected[:, p1],
                                           projected[:, p2],
                                           bins=30)
        H_T = H.T
        vmin = np.min(np.min(H_T))
        vmax = np.max(np.max(H_T))
        plt.pcolor(H_T, vmin=vmin, vmax=vmax)
        plt.xlabel("Principal Component %s" % p1, fontsize=16)
        plt.ylabel("Principal Component %s" % p2, fontsize=16)
        x_ticks = [round(f, 1) for f in xedges]
        y_ticks = [round(f, 1) for f in yedges]
        plt.xticks(np.arange(H_T.shape[0] + 1)[::2], x_ticks[::2])
        plt.yticks(np.arange(H_T.shape[1] + 1)[::2], y_ticks[::2])
        plt.xlim([0.0, H_T.shape[0]])
        plt.ylim([0.0, H_T.shape[1]])
        #plt.gca().invert_xaxis()
        #plt.gca().invert_yaxis()
        plt.tight_layout()
        #plt.colorbar()
        
        fig_flname = os.path.join(args.figures_dir,
                                  "pca_projection_%s_%s.png" % (str(p1), str(p2)))
        plt.savefig(fig_flname,
                    DPI=300)

    
def parseargs():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    comp_parser = subparsers.add_parser("compute-pca",
                                        help="Compute PCA")

    comp_parser.add_argument("--n-components",
                             type=int,
                             required=True,
                             help="Number of PCs to compute")

    comp_parser.add_argument("--pdb-file",
                             type=str,
                             required=True,
                             help="Input PDB file")

    comp_parser.add_argument("--input-traj",
                             type=str,
                             required=True,
                             help="Input trajectory file")

    comp_parser.add_argument("--model-file",
                             type=str,
                             required=True,
                             help="File to which to save model")

    comp_dist_parser = subparsers.add_parser("compute-dist-pca",
                                             help="Compute PCA")

    comp_dist_parser.add_argument("--n-components",
                                  type=int,
                                  required=True,
                                  help="Number of PCs to compute")

    comp_dist_parser.add_argument("--pdb-file",
                                  type=str,
                                  required=True,
                                  help="Input PDB file")

    comp_dist_parser.add_argument("--input-traj",
                                  type=str,
                                  required=True,
                                  help="Input trajectory file")

    comp_dist_parser.add_argument("--model-file",
                                  type=str,
                                  required=True,
                                  help="File to which to save model")
    
    eva_parser = subparsers.add_parser("explained-variance-analysis",
                                       help="Plot explained variances of PCs")

    eva_parser.add_argument("--figures-dir",
                            type=str,
                            required=True,
                            help="Figure output directory")

    eva_parser.add_argument("--model-file",
                            type=str,
                            required=True,
                            help="File from which to load model")
    
    proj_parser = subparsers.add_parser("plot-projections",
                                        help="Plot structures onto projections")

    proj_parser.add_argument("--figures-dir",
                             type=str,
                             required=True,
                             help="Figure output directory")
    
    proj_parser.add_argument("--pairs",
                             type=int,
                             nargs="+",
                             required=True,
                             help="Pairs of PCs to plot")

    proj_parser.add_argument("--model-file",
                             type=str,
                             required=True,
                             help="File from which to load model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if args.mode == "compute-pca":
        compute_pca(args)
    elif args.mode == "compute-dist-pca":
        compute_distance_pca(args)
    elif args.mode == "explained-variance-analysis":
        explained_variance_analysis(args)
    elif args.mode == "plot-projections":
        plot_projections(args)
    else:
        print "Unknown mode '%s'" % args.mode
        sys.exit(1)
