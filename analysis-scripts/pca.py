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
import matplotlib.patches as patches
import mdtraj as md
import numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from msmbuilder.decomposition import tICA
import numpy.linalg as LA

MODEL_TYPE_KEY = "model-type"
PCA_MODEL = "pca"
SVD_MODEL = "svd"
ICA_MODEL = "ica"
TICA_MODEL = "tica"
MODEL_KEY = "model"
PROJECTION_KEY = "projected-coordinates"
LAG_TIME_KEY = "lag-time"

def extract_features(args):
    print "reading trajectory"
    traj = md.load(args.input_traj,
                   top=args.pdb_file)

    if args.feature_type == "positions":
        print "aligning frames"
        traj.superpose(traj)

        features = traj.xyz.reshape(traj.n_frames,
                                    traj.n_atoms * 3)

    elif args.feature_type == "transformed-dihedrals":
        print "computing dihedrals"
        _, phi_angles = md.compute_phi(traj,
                                       periodic=False)
        _, psi_angles = md.compute_psi(traj,
                                       periodic=False)

        phi_sin = np.sin(phi_angles)
        phi_cos = np.cos(phi_angles)
        psi_sin = np.sin(psi_angles)
        psi_cos = np.cos(psi_angles)

        features = np.hstack([phi_sin,
                              phi_cos,
                              psi_sin,
                              psi_cos])

    elif args.feature_type == "transformed-dihedrals-chi":
        print "computing dihedrals"
        _, phi_angles = md.compute_phi(traj,
                                       periodic=False)
        _, psi_angles = md.compute_psi(traj,
                                       periodic=False)
        _, chi_angles = md.compute_chi1(traj,
                                        periodic=False)

        phi_sin = np.sin(phi_angles)
        phi_cos = np.cos(phi_angles)
        psi_sin = np.sin(psi_angles)
        psi_cos = np.cos(psi_angles)
        chi_sin = np.sin(chi_angles)
        chi_cos = np.cos(chi_angles)

        features = np.hstack([phi_sin,
                              phi_cos,
                              psi_sin,
                              psi_cos,
                              chi_sin,
                              chi_cos])

    else:
        raise Exception, "Unknown feature type '%s'", args.features

    return features

def train_model(args):
    features = extract_features(args)

    print "Fitting %s model" % args.model
    
    if args.model == "PCA":
        model = PCA(n_components = args.n_components)
        model_type = PCA_MODEL
        projected = model.fit_transform(features)

    elif args.model == "SVD":
        model = TruncatedSVD(n_components = args.n_components)
        model_type = SVD_MODEL
        projected = model.fit_transform(features)

    elif args.model == "ICA":
        model = FastICA(n_components = args.n_components)
        model_type = ICA_MODEL
        projected = model.fit_transform(features)

    elif args.model == "tICA":
        model = tICA(n_components = args.n_components,
                     kinetic_mapping=True,
                     lag_time = args.lag_time)
        model_type = TICA_MODEL
        projected = model.fit_transform([features])[0]

    else:
        raise Exception, "Unknown model type '%s'", args.model
    

    print "Writing model"
    model = { LAG_TIME_KEY : args.lag_time,
              MODEL_TYPE_KEY : model_type,
              MODEL_KEY : model,
              PROJECTION_KEY : projected }
    
    joblib.dump(model, args.model_file)

    
def explained_variance_analysis(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    data = joblib.load(args.model_file)
    model = data[MODEL_KEY]

    plt.clf()
    plt.grid(True)
    plt.plot(model.explained_variance_ratio_, "m.-")
    plt.xlabel("Principal Component", fontsize=16)
    plt.ylabel("Explained Variance Ratio", fontsize=16)
    plt.ylim([0., 1.])
    fig_flname = os.path.join(args.figures_dir, "pca_explained_variance_ratios.png")
    plt.savefig(fig_flname,
                DPI=300)

def timescale_analysis(args):
    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    data = joblib.load(args.model_file)
    if data[MODEL_TYPE_KEY] != TICA_MODEL:
        raise Exception, "Timescales can only be calculated for tICA"

    model = data[MODEL_KEY]
    lag_time = data[LAG_TIME_KEY]
    timescales = np.abs(model.timescales_ * args.timestep)

    for ts in timescales:
        plt.semilogy([0, 1],
                     [ts, ts],
                     "k-")

    plt.ylabel("Timescale (ns, log10)", fontsize=16)
    plt.xlim([0., 1.])
    plt.ylim([np.power(10., np.floor(min(np.log10(timescales)))),
              np.power(10., np.ceil(max(np.log10(timescales))))])
    fig_flname = os.path.join(args.figures_dir, "timescales.png")
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
        plt.xlabel("Component %s" % p1, fontsize=16)
        plt.ylabel("Component %s" % p2, fontsize=16)
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

def plot_projected_timeseries(args):
    model = joblib.load(args.model_file)
    projected = model[PROJECTION_KEY]

    for dim in args.dimensions:
        plt.plot(projected[:, dim],
                 label=str(dim))
        plt.xlabel("Time (frames)", fontsize=16)
        plt.ylabel("Projected Value", fontsize=16)
        plt.tight_layout()
        plt.legend()

    fig_flname = os.path.join(args.figures_dir,
                              "projected_timeseries")
    for dim in args.dimensions:
        fig_flname += "_%s" % dim
    fig_flname += ".png"

    plt.savefig(fig_flname,
                DPI=300)

def calculate_magnitudes(args):
    model = joblib.load(args.model_file)
    svd = model[MODEL_KEY]

    n_dim = svd.components_.shape[0]
    # really n_residues - 1
    n_residues = svd.components_.shape[1] / 4
    components = svd.components_.reshape(n_dim, n_residues, -1)
    
    # y dim should be phi_sin, phi_cos, psi_sin, psi_cos
    # first residue has no phi angle
    # last residue has no psi angle
    # so we only have pairs for residues 1 to n - 2
    components = np.stack([components[:, :-1, 0],
                           components[:, :-1, 1],
                           components[:, 1:, 2],
                           components[:, 1:, 3]],
                          axis=2)
    
    resids = range(2, n_residues + 1)
    for dim in args.dimensions:
        magnitudes = np.sqrt(np.sum(components[dim, :, :]**2, axis=1))
    
        if args.figures_dir:
            plt.clf()
            plt.plot(resids,
                     magnitudes,
                     label=str(dim))

            height = 0.1

            binding = patches.Rectangle((71, -0.1),
                                        263 - 70,
                                        height,
                                        linewidth=1,
                                        edgecolor='y',
                                        facecolor='y')
    
            tm1 = patches.Rectangle((264, -0.1),
                                    286 - 264,
                                    height,
                                    linewidth=1,
                                    edgecolor='m',
                                    facecolor='m')

            tm2 = patches.Rectangle((295, -0.1),
                                    317 - 295,
                                    height,
                                    linewidth=1,
                                    edgecolor='m',
                                    facecolor='m')

            tm3 = patches.Rectangle((327, -0.1),
                                    349 - 327,
                                    height,
                                    linewidth=1,
                                    edgecolor='m',
                                    facecolor='m')

            tm4 = patches.Rectangle((520, -0.1),
                                    537 - 520,
                                    height,
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
            plt.ylabel("Magnitude", fontsize=16)
            plt.tight_layout()
            
            fig_flname = os.path.join(args.figures_dir,
                                      "component_magnitudes_%s.png" % str(dim))
            
            plt.savefig(fig_flname,
                        DPI=300)

        if args.tsv_dir:
            flname = os.path.join(args.tsv_dir,
                                  "component_magnitudes_%s.tsv" % str(dim))
            with open(flname, "w") as fl:
                fl.write("residue\tmagnitude\n")
                pairs = zip(resids, magnitudes)
                pairs.sort(key=lambda p: p[1], reverse=True)
                for resid, magnitude in pairs:
                    fl.write("%s\t%s\n" % (resid, magnitude))

    
def parseargs():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    comp_parser = subparsers.add_parser("train-model",
                                        help="Train model")

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

    comp_parser.add_argument("--feature-type",
                             type=str,
                             required=True,
                             choices=["positions",
                                      "transformed-dihedrals",
                                      "transformed-dihedrals-chi"],
                             help="feature-type")

    comp_parser.add_argument("--model",
                             type=str,
                             required=True,
                             choices=["PCA",
                                      "SVD",
                                      "ICA",
                                      "tICA"],
                             help="model type")
    
    comp_parser.add_argument("--lag-time",
                             type=int,
                             default=1,
                             help="Subsample trajectory")
    
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

    ts_parser = subparsers.add_parser("timescale-analysis",
                                      help="Plot tICA timescales")

    ts_parser.add_argument("--figures-dir",
                           type=str,
                           required=True,
                           help="Figure output directory")

    ts_parser.add_argument("--model-file",
                           type=str,
                           required=True,
                           help="File from which to load model")

    ts_parser.add_argument("--timestep",
                           type=float,
                           required=True,
                           help="Elapsed time between frames")
    
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

    proj_ts_parser = subparsers.add_parser("plot-projected-timeseries",
                                           help="Plot projections over time")

    proj_ts_parser.add_argument("--figures-dir",
                                type=str,
                                required=True,
                                help="Figure output directory")
    
    proj_ts_parser.add_argument("--dimensions",
                                type=int,
                                nargs="+",
                                required=True,
                                help="Dimensions to plot")

    proj_ts_parser.add_argument("--model-file",
                                type=str,
                                required=True,
                                help="File from which to load model")

    comp_res_magn_parser = subparsers.add_parser("component-residue-magnitudes",
                                           help="Calculate component residue magnitudes")

    comp_res_magn_parser.add_argument("--figures-dir",
                                type=str,
                                help="Figure output directory")

    comp_res_magn_parser.add_argument("--tsv-dir",
                                type=str,
                                help="tsv output directory")
    
    comp_res_magn_parser.add_argument("--dimensions",
                                type=int,
                                nargs="+",
                                required=True,
                                help="Dimensions to plot")

    comp_res_magn_parser.add_argument("--model-file",
                                type=str,
                                required=True,
                                help="File from which to load model")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if args.mode == "train-model":
        train_model(args)
    elif args.mode == "explained-variance-analysis":
        explained_variance_analysis(args)
    elif args.mode =="timescale-analysis":
        timescale_analysis(args)
    elif args.mode == "plot-projections":
        plot_projections(args)
    elif args.mode == "plot-projected-timeseries":
        plot_projected_timeseries(args)
    elif args.mode == "component-residue-magnitudes":
        calculate_magnitudes(args)
    else:
        print "Unknown mode '%s'" % args.mode
        sys.exit(1)
