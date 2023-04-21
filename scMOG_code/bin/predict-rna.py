"""
Code to predict rna
"""

import os
import sys
import logging
import argparse
import copy
import functools
import itertools

import numpy as np
import pandas as pd
import scipy.spatial
import scanpy as sc
import collections
import matplotlib.pyplot as plt
from skorch.helper import predefined_split

import torch
import torch.nn as nn
import torch.utils.data as Data

from astropy.visualization.mpl_normalize import ImageNormalize
import random
from astropy.visualization import LogStretch
from typing import *



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scMOG"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)


import anndata as ad
import loss_functions
import activations
import utils
import sklearn.metrics as metrics
import both_GAN_1

logging.basicConfig(level=logging.INFO)



SAVEFIG_DPI = 1200

def build_parser():
    """Building a parameter parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--hidden", type=int, nargs="*", default=[16], help="Hidden dimensions"
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=[0.0001], nargs="*", help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, nargs="*", default=[512], help="Batch size"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=[182822], help="Random seed to use"
    )
    parser.add_argument("--device", default=0, type=int, help="Device to train on")
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    #parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    #parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    return parser


def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy's sparse matrix to torch's sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def plot_scatter_with_r(
    x: Union[np.ndarray, scipy.sparse.csr_matrix],
    y: Union[np.ndarray, scipy.sparse.csr_matrix],
    color=None,
    subset: int = 0,
    logscale: bool = False,
    density_heatmap: bool = False,
    density_dpi: int = 150,
    density_logstretch: int = 1000,
    title: str = "",
    xlabel: str = "Original norm counts",
    ylabel: str = "Inferred norm counts",
    xlim: Tuple[int, int] = None,
    ylim: Tuple[int, int] = None,
    one_to_one: bool = False,
    corr_func: Callable = scipy.stats.pearsonr,
    figsize: Tuple[float, float] = (7, 5),
    fname: str = "",
    ax=None,
):
    """
    Plot the given x y coordinates, appending Pearsons r
    Setting xlim/ylim will affect both plot and R2 calculation
    In other words, plot view mirrors the range for which correlation is calculated
    """
    assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
    if color is not None:
        assert color.size == x.size
    if one_to_one and (xlim is not None or ylim is not None):
        assert xlim == ylim
    if xlim:
        keep_idx = utils.ensure_arr((x >= xlim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    if ylim:
        keep_idx = utils.ensure_arr((y >= ylim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    # x and y may or may not be sparse at this point
    assert x.shape == y.shape
    if subset > 0 and subset < x.size:
        logging.info(f"Subsetting to {subset} points")
        random.seed(1234)
        # Converts flat index to coordinates
        indices = np.unravel_index(
            np.array(random.sample(range(np.product(x.shape)), k=subset)), shape=x.shape
        )
        x = utils.ensure_arr(x[indices])
        y = utils.ensure_arr(y[indices])
        if isinstance(color, (tuple, list, np.ndarray)):
            color = np.array([color[i] for i in indices])

    if logscale:
        x = np.log1p(x.cpu())
        y = np.log1p(y.cpu())

    # Ensure correct format
    x = ensure_arr(x).flatten()
    y = ensure_arr(y).flatten()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    logging.info(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    # spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    # logging.info(
    #     f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}"
    # )

    if ax is None:
        fig = plt.figure(dpi=300, figsize=figsize)
        if density_heatmap:
            # https://github.com/astrofrog/mpl-scatter-density
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    if density_heatmap:
        norm = None
        if density_logstretch:
            norm = ImageNormalize(
                vmin=0, vmax=100, stretch=LogStretch(a=density_logstretch)
            )
        ax.scatter_density(x, y, dpi=density_dpi, norm=norm, color="tab:blue")
    else:
        ax.scatter(x, y, alpha=0.2, c=color)

    if one_to_one:
        unit = np.linspace(*ax.get_xlim())
        ax.plot(unit, unit, linestyle="--", alpha=0.5, label="$y=x$", color="grey")
        ax.legend()
    ax.set(
        xlabel=xlabel + (" (log)" if logscale else ""),
        ylabel=ylabel + (" (log)" if logscale else ""),
        title=(title + f" ($r={pearson_r:.2f}$)").strip(),
    )
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)

    if fig is not None and fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")

    return fig



def rmse_value(truth,
               preds,
):
    'Calculate RMSE'
    truth = truth.flatten()
    preds = preds.flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")


def main():
    """Run Script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")



    sc_rna_dataset=ad.read_h5ad('truth_rna_GM.h5ad')
    sc_atac_dataset=ad.read_h5ad('truth_atac_GM.h5ad')

    cuda = True if torch.cuda.is_available() else False
# Model
    param_combos = list(
        itertools.product(
            args.hidden, args.lr, args.seed
        )
    )
    for h_dim, lr, rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lr_{lr}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)

        generator = both_GAN_1.GeneratorRNA(hidden_dim=h_dim,
                                               input_dim1=sc_rna_dataset.X.shape[1],
                                               input_dim2=sc_atac_dataset.X.shape[1],
                                               # out_dim=get_per_chrom_feature_count(sc_atac_dataset),
                                               final_activations1=[activations.Exp(), activations.ClippedSoftplus()],
                                               flat_mode=True,
                                               seed=rand_seed,
                                               )

        if cuda:
           generator.cuda()


        #ATAC->RNA

        logging.info("Evaluating ATAC > RNA")
        sc_rna_test_dataset = ad.read_h5ad('truth_rna_GM.h5ad')
        Acoo = sc_rna_test_dataset.X.tocoo()
        sc_rna_test = scipy_sparse_mat_to_torch_sparse_tensor(Acoo)
        sc_rna_test = sc_rna_test.to_dense()

        sc_atac_test_dataset = ad.read_h5ad('truth_atac_GM.h5ad')
        Acoo1 = sc_atac_test_dataset.X.tocoo()
        sc_atac_test= scipy_sparse_mat_to_torch_sparse_tensor(Acoo1)
        sc_atac_test = sc_atac_test.to_dense()



        test_iter = torch.utils.data.DataLoader(dataset=sc_atac_test, batch_size=64 )
        def pridect(test_iter):
           generator.eval()
           first=1
           generator.load_state_dict(torch.load('RNAgenerator.pth',map_location='cpu'))
           for x in test_iter:
              if cuda:
                   x = x.cuda()
              with torch.no_grad():  # 禁止梯度的计算
                  y_pred = generator(x)
                  if first==1:
                      ret=y_pred[0]
                      first=0
                  else:
                      ret=torch.cat((ret,y_pred[0]),0)
           return ret

        sc_atac_rna_test_preds =pridect(test_iter)


        sc_atac_rna_full_preds_anndata = sc.AnnData(
            scipy.sparse.csr_matrix(sc_atac_rna_test_preds),
            obs=sc_atac_test_dataset.obs.copy(deep=True),
        )
        sc_atac_rna_full_preds_anndata.var_names =sc_rna_dataset.var_names
        logging.info("Writing RNA from ATAC")

        # Seurat also expects the raw attribute to be populated
        sc_atac_rna_full_preds_anndata.raw = sc_atac_rna_full_preds_anndata.copy()
        sc_atac_rna_full_preds_anndata.write(
            os.path.join(args.outdir, f"atac_rna_adata.h5ad")
        )

        fig = plot_scatter_with_r(
            sc_rna_test,
            sc_atac_rna_test_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title="scMOG ATAC > RNA",
            fname=os.path.join(outdir_name, f"mymodel atac_rna_scatter_log_GM_AE.{args.ext}"),
        )
        plt.close(fig)
        # fig1 = plot_prc(
        #     sc_atac_test,
        #     babel_sc_atac_test,
        #     title_prefix="RNA > ATAC",
        #     fname=os.path.join(outdir_name, f"rna_atac_auprc.{args.ext}"),
        # )
        # rmse_value(sc_atac_test, babel_sc_atac_test)





if __name__ == "__main__":
    main()