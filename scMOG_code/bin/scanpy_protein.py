#Protein downstream analysis

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
params = {
       'legend.fontsize': 'x-large','axes.labelsize': 'x-large'}
#params = {'axes.labelsize': 16,
#          'axes.titlesize': 16}
plt.rcParams.update(params)


adata = ad.read_h5ad("turth_protein.h5ad")

adata.var_names_make_unique()
print(adata)

adata.var= pd.DataFrame(index=['CD3', 'CD4', 'CD8a', 'CD14',
       'CD15', 'CD16', 'CD56', 'CD19',
       'CD25', 'CD45RA', 'CD45RO',
       'PD-1', 'TIGIT', 'CD127',
       'IgG2a', 'IgG1',
       'IgG2b'])
adata.raw=adata.copy()

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

# sc.tl.leiden(adata,key_added='clusters', resolution=0.7)
sc.tl.leiden(adata,key_added='clusters',resolution=0.7 )
print(adata)
sc.set_figure_params(scanpy=True, fontsize=15)
sc.pl.umap(adata, color='clusters',legend_loc='on data',)


adata_predict = ad.read_h5ad("RNA_protein_adata.h5ad")
adata_predict.var=adata.var
adata_predict.obs=adata.obs
adata_predict.raw=adata_predict.copy()



# sc.set_figure_params(scanpy=True, fontsize=22)
sc.pl.heatmap(adata=adata,var_names=adata.var_names,groupby='clusters', cmap='viridis',vmin=-4,vmax=6,)


sc.pl.heatmap(adata_predict, adata_predict.var_names, groupby='clusters',cmap='viridis',vmin=-4,vmax=6,)

# For example, imputed CD4 and CD8 levels separate CD4+ T cells
# from CD8+ T cells with high confidence. Further separation of
# naïve T cells to memory T cells can be achieved through imputed
# CD45RA/CD45RO abundance, as CD45RA is a naïve antigen and
# CD45RO is a memory antigen.


# sc.pl.heatmap(adata, adata.var_names, groupby='leiden', cmap='viridis',vmin=-8,vmax=10)
# sc.pl.heatmap(adata_predict, adata_predict.var_names, groupby='leiden', cmap='viridis',vmin=-8,vmax=10)
sc.pl.violin(adata,['CD14','CD16',], groupby='clusters',)
sc.pl.violin(adata_predict,['CD14','CD16'], groupby='clusters',)

sc.pl.violin(adata,['CD19','CD8a'], groupby='clusters',)
sc.pl.violin(adata_predict,['CD19','CD8a'], groupby='clusters',)

sc.pl.violin(adata,['CD45RA','CD45RO','CD4'], groupby='clusters',)
sc.pl.violin(adata_predict,['CD45RA','CD45RO','CD4'], groupby='clusters',)

