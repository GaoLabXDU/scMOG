#Convert h5 file to mtx
import os
import sys

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scMOG",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)


import scanpy as sc
import adata_utils

def sc_read_10x_h5_ft_type(fname: str, ft_type: str) :
    """Read the h5 file, taking only features with specified ft_type"""
    assert fname.endswith(".h5")

    parsed = sc.read_10x_h5(fname, gex_only=False)
    parsed.var_names_make_unique()
    assert ft_type in set(
        parsed.var["feature_types"]
    ), f"Given feature type {ft_type} not in included types: {set(parsed.var['feature_types'])}"

    retval = parsed[
        :,
        [n for n in parsed.var_names if parsed.var.loc[n, "feature_types"] == ft_type],
    ]
    return retval

x = sc_read_10x_h5_ft_type('pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5', ft_type="Antibody Capture")

y=sc_read_10x_h5_ft_type('5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5', ft_type="Antibody Capture")

# y.var_names=Series(['CD3', 'CD4', 'CD8a', 'CD11b',
#        'CD14', 'CD15_TotalSeqB', 'CD16', 'CD19',
#        'CD20', 'CD25', 'CD27', 'CD28',
#        'CD34', 'CD45RA', 'CD45RO',
#        'CD56', 'CD62L', 'CD69', 'CD80',
#        'CD86', 'CD127', 'CD137',
#        'CD197', 'CD274', 'CD278',
#        'CD335', 'PD-1', 'HLA-DR_TotalSeqB',
#        'TIGIT', 'IgG1', 'IgG2a',
#        'IgG2b'])
# retval = y[:,[n for n in y.var_names if n in x.var_names]]

retval=y[:,x.var_names]


adata_utils.write_adata_as_10x_dir(x, outdir='CITE_train_10k_protein',mode='RNA')
adata_utils.write_adata_as_10x_dir(retval, outdir='CITE_test_5k_protein',mode='RNA')



x = sc_read_10x_h5_ft_type('pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5', ft_type="Gene Expression")
adata_utils.write_adata_as_10x_dir(x, outdir='CITE_train_10k_RNA',mode='RNA')

y=sc_read_10x_h5_ft_type('5k_pbmc_protein_v3_nextgem_filtered_feature_bc_matrix.h5', ft_type="Gene Expression")
retval=y[:,x.var_names]
adata_utils.write_adata_as_10x_dir(retval, outdir='CITE_test_5k_RNA',mode='RNA')


# retval = y[:,[n for n in y.var_names if n in x.var_names]]
# print(retval)

# parsed = sc.read_10x_h5('pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5', gex_only=False)
# print(parsed)
# print(parsed.var["feature_types"])