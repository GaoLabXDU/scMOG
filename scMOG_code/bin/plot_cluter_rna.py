#RNA Downstream Analysis

import os
import sys
import importlib


import matplotlib.pyplot as plt

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scMOG",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import numpy as np
import pandas as pd
import anndata as ad
import adata_utils
import plot_utils
import utils

outdir_name='mymodel_polt'
if not os.path.isdir(outdir_name):
    assert not os.path.exists(outdir_name)
    os.makedirs(outdir_name)
assert os.path.isdir(outdir_name)

pbmc_seurat_atac_obs = pd.read_csv("pbmc_atac_obs_metadata.csv", index_col=0)
pbmc_rna = ad.read_h5ad("atac_rna_adata.h5ad")

if not isinstance(pbmc_rna.X, np.ndarray):
    pbmc_rna.X = pbmc_rna.X.toarray()
#Subset to the cells that we have Signac annotations for
intersected_barcodes = [bc for bc in pbmc_rna.obs_names if bc in pbmc_seurat_atac_obs.index]
pbmc_rna = pbmc_rna[intersected_barcodes]
pbmc_rna.obs['seurat_clustername'] = pbmc_seurat_atac_obs['ClusterName']
# # Compute size-normalized, log-normalized counts
pbmc_rna_log = adata_utils.normalize_count_table(pbmc_rna, size_factors=True, log_trans=True, normalize=False)

# Perform dimensionality reduction and clustering
plot_utils.preprocess_anndata(pbmc_rna_log, louvain_resolution=0, leiden_resolution=0.8)
#
pbmc_rna_log.write_h5ad(
    os.path.join(outdir_name, "mymodel_pbmc_rna_log.h5ad"))

pbmc_rna_log = ad.read_h5ad("mymodel_polt/mymodel_pbmc_rna_log.h5ad")



importlib.reload(plot_utils)

def celltype_to_color_atac_base(celltype:str):
    """Assign each celltype a consistent, unique color"""
    tab20 = plt.get_cmap("tab20")
    mapping = {
        "CD14 Mono": tab20(2),
        "CD16 Mono": tab20(11),
        "CD4 Memory": tab20(4),
        "CD4 Naive": tab20(5),
        "CD8 Effector": tab20(19),
        "CD8 Naive": tab20(18),
        "DC": tab20(6),
        "DN T": tab20(16),
        "NK CD56Dim": tab20(8),
        "NK CD56bright": tab20(9),
        "pDC": tab20(7),
        "pre-B": tab20(0),
        "pro-B": tab20(1),
    }
    rgb = dict(enumerate(mapping.values()))[celltype][:3]  # Drops alpha transparency
    if np.any(np.array(rgb) > 1):  # mpl does RGB in float
        rgb = np.array(rgb) / 255
    return tuple(rgb)

plot_utils.plot_clustering_anndata_direct_label(
    pbmc_rna_log,
    color="seurat_clustername",
    cmap=celltype_to_color_atac_base,
    adjust=True,
    title="BABEL",
    fname=os.path.join(outdir_name, f"rna_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "NKG7",
    title="NKG7 (NK cells,BABEL)",
    # cbar_pos=[0.16, 0.83, 0.28, 0.03],
fname=os.path.join(outdir_name, f"NKG7_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "FCER1A",
    title="FCER1A (Dendritic Cells,BABEL)",
    # cbar_pos=[0.16, 0.83, 0.28, 0.03],
fname=os.path.join(outdir_name, f"FCER1A_cluster.pdf"),
).show()


plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "CST3",
    title="CST3 (Dendritic Cells,BABEL)",
    # cbar_pos=[0.16, 0.83, 0.28, 0.03],
fname=os.path.join(outdir_name, f"CST3_cluster.pdf"),
).show()
plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "CD14",
    title="CD14 (CD14+ mono,BABEL)",
    cbar_pos=[0.16, 0.83, 0.28, 0.03],
fname=os.path.join(outdir_name, f"CD14_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "MS4A1",
    title="MS4A1 (B cells,BABEL)",
fname=os.path.join(outdir_name, f"MS4A1_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "LYZ",
    title="LYZ (CD14+ mono,BABEL)",
fname=os.path.join(outdir_name, f"LYZ_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "GNLY",
    title="GNLY (NK cells,BABEL)",
fname=os.path.join(outdir_name, f"GNLY_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "IL7R",
    title="IL7R (CD4 T cells,BABEL)",
fname=os.path.join(outdir_name, f"IL7R_cluster.pdf"),
).show()

plot_utils.plot_clustering_anndata_gene_color(
    pbmc_rna_log,
    "CD8A",
    title="CD8A (CD8 T cells,BABEL)",
fname=os.path.join(outdir_name, f"CD8A_cluster.pdf"),
).show()
