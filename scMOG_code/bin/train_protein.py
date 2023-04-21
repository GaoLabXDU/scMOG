"""
Code to train a model
"""

import os
import sys
import logging
import argparse

import itertools

import numpy as np
import pandas as pd
import scipy.spatial
import scanpy as sc

import matplotlib.pyplot as plt
from anndata import AnnData
import torch
import torch.nn as nn
import torch.utils.data as Data
import sklearn.metrics as metrics
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

import both_GAN_1
from datasets_CITE import RNA_Dataset, ATAC_Dataset
import utils
import lossfunction
import losses
from pytorchtools import EarlyStopping


logging.basicConfig(level=logging.INFO)

SAVEFIG_DPI = 1200

def build_parser():
    """"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset_path', type=str, default="")

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
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    return parser

def normalize_count_table(
    x: AnnData,
    size_factors: bool = True,
    log_trans: bool = True,
    normalize: bool = True,
) -> AnnData:
    """
    Normalize the count table using method described in DCA paper, performing operations IN PLACE
    rows correspond to cells, columns correspond to genes (n_obs x n_vars)
    s_i is the size factor per cell, total number of counts per cell divided by median of total counts per cell
    x_norm = zscore(log(diag(s_i)^-1 X + 1))

    Reference:
    https://github.com/theislab/dca/blob/master/dca/io.py

    size_factors - calculate and normalize by size factors
    top_n - retain only the top n features with largest variance after size factor normalization
    normalize - zero mean and unit variance
    log_trans - log1p scale data
    """
    assert isinstance(x, AnnData)
    if log_trans or size_factors or normalize:
        x.raw = x.copy()  # Store the original counts as .raw
    # else:
    #     x.raw = x
    if size_factors:
        logging.info("Computing size factors")
        n_counts = np.squeeze(
            np.array(x.X.sum(axis=1))
        )  # Number of total counts per cell
        # Normalizes each cell to total count equal to the median of total counts pre-normalization
        sc.pp.normalize_total(x, inplace=True)
        # The normalized values multiplied by the size factors give the original counts
        x.obs["size_factors"] = n_counts / np.median(n_counts)
        x.uns["median_counts"] = np.median(n_counts)
        logging.info(f"Found median counts of {x.uns['median_counts']}")
        logging.info(f"Found maximum counts of {np.max(n_counts)}")
    else:
        x.obs["size_factors"] = 1.0
        x.uns["median_counts"] = 1.0

    if log_trans:  # Natural logrithm
        logging.info("Log transforming data")
        sc.pp.log1p(
            x,
            chunked=True,
            copy=False,
            chunk_size=100000,
        )

    if normalize:
        logging.info("Normalizing data to zero mean unit variance")
        sc.pp.scale(x, zero_center=True, copy=False)

    return x


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

def clr_transform(x: np.ndarray, add_pseudocount: bool = True) -> np.ndarray:
    """
    Centered logratio transformation. Useful for protein data, but

    >>> clr_transform(np.array([0.1, 0.3, 0.4, 0.2]), add_pseudocount=False)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])
    >>> clr_transform(np.array([[0.1, 0.3, 0.4, 0.2], [0.1, 0.3, 0.4, 0.2]]), add_pseudocount=False)
    array([[-0.79451346,  0.30409883,  0.5917809 , -0.10136628],
           [-0.79451346,  0.30409883,  0.5917809 , -0.10136628]])
    """
    assert isinstance(x, np.ndarray)
    if add_pseudocount:
        x = x + 1.0
    if len(x.shape) == 1:
        denom = scipy.stats.mstats.gmean(x)
        retval = np.log(x / denom)
    elif len(x.shape) == 2:
        # Assumes that each row is an independent observation
        # and that columns denote features
        per_row = []
        for i in range(x.shape[0]):
            denom = scipy.stats.mstats.gmean(x[i])
            row = np.log(x[i] / denom)
            per_row.append(row)
        assert len(per_row) == x.shape[0]
        retval = np.stack(per_row)
        assert retval.shape == x.shape
    else:
        raise ValueError(f"Cannot CLR transform array with {len(x.shape)} dims")
    return retval

def plot_loss_history(history1,history2,history3,fname: str):
    """Constructing training loss curves"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(
        np.arange(len(history1)), history1, label="Train_G",
    )
    if len(history2):
        ax.plot(
        np.arange(len(history2)), history2, label="Train_D",)
    if len(history3):
        ax.plot(
        np.arange(len(history3)), history3, label="Test_G",)
    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel="Loss",
    )
    #plt.show()
    fig.savefig(fname)
    return fig


def rmse_value(truth,
               preds,
):
    'Calculate RMSE'
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")

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
    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    logging.info(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    logging.info(
        f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}"
    )

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




def main():
    """train"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    dataset_path = args.dataset_path
    # rna_path = dataset_path + '/RNA-seq'
    # modal_path = dataset_path + '/{}'.format('CITE-seq')
    rna_path = dataset_path + '/CITE_train_10k_RNA'
    modal_path = dataset_path + '/CITE_train_10k_protein'

    r_dataset = RNA_Dataset(rna_path)
    modal_dataset =RNA_Dataset(modal_path)

    rna_test_path = dataset_path + '/CITE_test_5k_RNA'
    modal_test_path = dataset_path + '/CITE_test_5k_protein'
    test_r_dataset = RNA_Dataset(rna_test_path)
    test_modal_dataset =RNA_Dataset(modal_test_path)

    # print("RNA-seq shape is " + str(r_dataset.data.shape))
    # print("{} shape is ".format('CITE-seq') + str(modal_dataset.data.shape))
    # print("RNA-seq shape is " + str( test_r_dataset.data.shape))
    # print("{} shape is ".format('CITE-seq') + str(test_modal_dataset.data.shape))

    # Split train test
    np.random.seed(18823)

    num_cell = r_dataset.data.shape[0]
    t_size = np.round(num_cell * 1).astype('int')
    t_id = np.random.choice(a=num_cell, size=t_size, replace=False)
    s_id = np.delete(range(num_cell), t_id)


    cuda = True if torch.cuda.is_available() else False
    device_ids = range(torch.cuda.device_count())

# Model
    param_combos = list(
        itertools.product(
            args.hidden, args.lr,args.seed
        )
    )
    for h_dim,lr,rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lr_{lr}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)

        torch.save(t_id, os.path.join(outdir_name, f"t_id.rar"))
        torch.save(s_id, os.path.join(outdir_name, f"s_id.rar"))



        GeneratorProtein = both_GAN_1.GeneratorProtein(hidden_dim=h_dim,
                                           input_dim1=r_dataset.data.shape[1],
                                           input_dim2=modal_dataset.data.shape[1],
                                           # out_dim=get_per_chrom_feature_count(sc_atac_dataset),
                                           final_activations2=nn.Identity(),
                                           flat_mode=True,
                                           seed=rand_seed,
                                           )


        DiscriminatorProtein = both_GAN_1.DiscriminatorProtein(input_dim=modal_dataset.data.shape[1],seed=rand_seed)


        # Loss function
        loss_rna = lossfunction.loss
        loss_protein=nn.MSELoss()

        def loss_D(fake,real,Discriminator):
            loss2_1 = -torch.mean(Discriminator(real))
            if isinstance(fake, tuple):
                loss2_2 = torch.mean(Discriminator(fake[0].detach()))
            else:
                loss2_2 = torch.mean(Discriminator(fake.detach()))
            loss2 = loss2_1 + loss2_2
            return loss2

        def loss_rna_G(fake,Discriminator):
            loss1 =-torch.mean(Discriminator(fake[0]))
            return loss1

        def loss_atac_G(fake,Discriminator):
            loss1 = -torch.mean(Discriminator(fake))
            return loss1


        if cuda:
           GeneratorProtein.cuda()
           DiscriminatorProtein.cuda()


        if len(device_ids) > 1:
            GeneratorProtein = torch.nn.DataParallel(GeneratorProtein)
            DiscriminatorProtein = torch.nn.DataParallel(DiscriminatorProtein)


        optimizer_protein_1 = torch.optim.Adam(GeneratorProtein.parameters(), lr=lr, betas=(args.b1, args.b2))
        optimizer_protein = torch.optim.RMSprop(GeneratorProtein.parameters(), lr=lr)
        optimizer_D_protein = torch.optim.RMSprop(DiscriminatorProtein.parameters(), lr=lr)

        def pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history):
            generator.train()
            discriminator.train()
            train_losses=[]
            trainD_losses=[]
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterD.zero_grad()
                y_fake = generator(x)
                loss2=loss_D(y_fake,y,discriminator)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())

                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

                if i % 5 == 0:
                    updaterG.zero_grad()
                    y_hat =generator(x)
                    if isinstance(y_hat, tuple):
                        loss1=loss_rna_G(y_hat,discriminator)
                    else:
                        loss1 = loss_atac_G(y_hat,discriminator)
                    loss1.backward()
                    updaterG.step()
                    train_losses.append(loss1.item())

            train_loss = np.average(train_losses[:-1])
            trainD_loss = np.average(trainD_losses[:-1])
            logging.info(f"lossG: {train_loss}")
            logging.info(f"lossD: {trainD_loss}")
            lossG_history.append(train_loss)
            lossD_history.append(trainD_loss)

            return lossG_history, lossD_history

        def training_epoch(train_iter, generator, updaterG,lossG_history):
            generator.train()
            train_losses = []
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterG.zero_grad()
                y_hat = generator(x)
                if isinstance(y_hat, tuple):
                    loss=loss_rna(preds=y_hat[0],theta=y_hat[1],truth=y)
                else:
                    loss = loss_protein(y_hat,y)
                loss.backward()
                updaterG.step()
                train_losses.append(loss.item())
            train_loss = np.average(train_losses[:-1])
            logging.info(f"AEloss: {train_loss}")
            lossG_history.append(train_loss)
            return lossG_history

        def test_epoch(generator,discriminator,test_iter,lossG_test_history):
            generator.eval()
            if discriminator:
                discriminator.eval()
            valid_losses = []
            with torch.no_grad():
                for (x,y)in test_iter:
                    if cuda:
                        x= x.cuda()
                        y= y.cuda()
                    y_hat =generator(x)
                    if discriminator:
                        if isinstance(y_hat, tuple):
                            loss=loss_rna_G(y_hat,discriminator)
                        else:
                            loss= loss_atac_G(y_hat,discriminator)
                    else:
                        if isinstance(y_hat, tuple):
                            loss = loss_rna(preds=y_hat[0], theta=y_hat[1], truth=y)
                        else:
                            loss = loss_protein(y_hat, y)
                    valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses[:-1])
            logging.info(f"loss_test: {valid_loss}")
            lossG_test_history.append(valid_loss)
            return lossG_test_history,valid_loss

        def predict_protein(truth,generator,truth_iter):
            logging.info("....................................Evaluating protein")

            def predict1(generator,truth_iter):
                generator.eval()
                first = 1
                for x in truth_iter:
                    if cuda:
                        x = x.cuda()
                    with torch.no_grad():  # 禁止梯度的计算
                        y_pred = generator(x)
                        if first == 1:
                            ret = y_pred
                            first = 0
                        else:
                            ret = torch.cat((ret, y_pred), 0)
                return ret

            sc_rna_protein_truth_preds = predict1(generator,truth_iter)


            fig = plot_scatter_with_r(
                truth,
                sc_rna_protein_truth_preds,
                one_to_one=True,
                # logscale=True,
                density_heatmap=True,
                title="RNA>protein (test set)",
                fname=os.path.join(outdir_name, f"RNA_protein_scatter_log_pre.{args.ext}"),
            )
            rmse_value(truth, sc_rna_protein_truth_preds)
            plt.close(fig)



        def train(generator,discriminator,num_epochs, train_iter,test_iter,truth_iter,truth,updaterG,updaterD,ISRNA):  # @save
            """"""
            lossG_history = []
            lossD_history = []
            lossG_test_history=[]

            early_stopping = EarlyStopping(patience=7,verbose=True)
            for epoch in range(num_epochs):
                logging.info(f"....................................................this is epoch: {epoch}")
                if discriminator:
                    lossG_history,lossD_history=pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history)
                    if ((epoch + 1) %7== 0):
                        predict_protein(truth, generator, truth_iter)
                else:
                    lossG_history=training_epoch(train_iter,generator,updaterG,lossG_history)
                    if ((epoch + 1) % 5== 0):
                        predict_protein(truth, generator, truth_iter)

                if test_iter:
                    lossG_test_history,lossG_test=test_epoch(generator,discriminator,test_iter,lossG_test_history)

                early_stopping(lossG_test, generator)

                if early_stopping.early_stop:
                    logging.info("early stopping")
                    predict_protein(truth, generator, truth_iter)
                    break

            return lossG_history, lossD_history,lossG_test_history



        #Dataset Summary
        var = pd.DataFrame(index=r_dataset.genes)
        obs = pd.DataFrame(index=r_dataset.barcode)

        sc_rna_full_anndata = sc.AnnData(
            r_dataset.data,
            obs=obs,
            var=var,
        )

        # sc_atac_rna_full_preds_anndata.var_names = var
        logging.info("Writing RNA")
        sc_rna_full_anndata  = normalize_count_table(  # Normalizes in place
            sc_rna_full_anndata ,
            size_factors=True,
            normalize=True,
            log_trans=True,
        )
        sc_rna_full_anndata.X= scipy.sparse.csr_matrix(sc_rna_full_anndata.X)
        # Seurat also expects the raw attribute to be populated
        sc_rna_full_anndata .raw = sc_rna_full_anndata .copy()
        sc_rna_full_anndata .write(
            os.path.join(args.outdir, f"full_rna.h5ad")
        )


        sc_rna_full_anndata.X= sc_rna_full_anndata.X.toarray()

        sc_rna = torch.Tensor(sc_rna_full_anndata.X)


        var = pd.DataFrame(index=test_r_dataset.genes)
        obs = pd.DataFrame(index=test_r_dataset.barcode)


        sc_rna_test_anndata = sc.AnnData(
            test_r_dataset.data,
            obs=obs,
            var=var,
        )

        # sc_atac_rna_full_preds_anndata.var_names = var
        logging.info("Writing RNA")
        sc_rna_test_anndata = normalize_count_table(  # Normalizes in place
            sc_rna_test_anndata,
            size_factors=True,
            normalize=True,
            log_trans=True,
        )
        sc_rna_test_anndata.X = scipy.sparse.csr_matrix(sc_rna_test_anndata.X)
        # Seurat also expects the raw attribute to be populated
        sc_rna_test_anndata.raw = sc_rna_test_anndata.copy()
        sc_rna_test_anndata.write(
            os.path.join(args.outdir, f"turth_rna.h5ad")
        )

        sc_rna_test_anndata.X = sc_rna_test_anndata.X.toarray()

        sc_rna_test = torch.Tensor(sc_rna_test_anndata.X)

        modal_dataset.data = utils.ensure_arr(modal_dataset.data)
        modal_dataset.data = clr_transform(modal_dataset.data)
        sc_protein=torch.Tensor(modal_dataset.data)

        test_modal_dataset.data = utils.ensure_arr(test_modal_dataset.data)
        test_modal_dataset.data = clr_transform(test_modal_dataset.data)
        sc_protein_test=torch.Tensor(test_modal_dataset.data)



        sc_rna_train=sc_rna
        sc_rna_test =sc_rna_test
        sc_protein_train=sc_protein
        sc_protein_test=sc_protein_test


        train_dataset1= Data.TensorDataset(sc_rna_train, sc_protein_train)
        train_iter1=torch.utils.data.DataLoader(dataset=train_dataset1,batch_size=256,shuffle=True)

        test_dataset1= Data.TensorDataset(sc_rna_test, sc_protein_test)
        test_iter1=torch.utils.data.DataLoader(dataset=test_dataset1,batch_size=128)


        truth_iter_rna = torch.utils.data.DataLoader(sc_rna_test, batch_size=64)

        var = pd.DataFrame(index=test_modal_dataset.genes)
        obs = pd.DataFrame(index=test_modal_dataset.barcode)
        sc_atac_rna_full_preds_anndata = sc.AnnData(
            scipy.sparse.csr_matrix(sc_protein_test),
            obs=obs,
            var=var,
        )

        logging.info("Writing protein from RNA")
        # Seurat also expects the raw attribute to be populated
        sc_atac_rna_full_preds_anndata.raw = sc_atac_rna_full_preds_anndata.copy()
        sc_atac_rna_full_preds_anndata.write(
            os.path.join(args.outdir, f"turth_protein.h5ad")
        )

        logging.info("...............................pretraining RNA -> protein")
        loss1_history, loss2_history, loss1_test_history = train(generator=GeneratorProtein,
                                                                 discriminator=DiscriminatorProtein, num_epochs=200,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_protein_test,
                                                                 updaterG=optimizer_protein,
                                                                 updaterD=optimizer_D_protein, ISRNA=False)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"losspretrain-protein.{args.ext}")
        )
        plt.close(fig)


        logging.info("........................................................................................................................................................")
        logging.info("training RNA ->protein")
        loss1_history, loss2_history, loss1_test_history = train(generator=GeneratorProtein,
                                                                 discriminator=None, num_epochs=120,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_protein_test,
                                                                 updaterG=optimizer_protein_1,
                                                                 updaterD=None, ISRNA=False)
        # loss visualization
        fig = plot_loss_history(
            loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossRNA-protein.{args.ext}")
        )
        plt.close(fig)



        ## save model

        torch.save(GeneratorProtein.state_dict(),os.path.join(outdir_name, f"Proteingenerator.pth"))
        torch.save(DiscriminatorProtein.state_dict(), os.path.join(outdir_name, f"Proteindiscriminator.pth"))




if __name__ == "__main__":
    main()