#Joint analysis through predicted ATAC data

library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(JASPAR2020)
library(TFBSTools)
library(BSgenome.Hsapiens.UCSC.hg38)
library(patchwork)


set.seed(1234)


features <- read.csv('predict_otherpbmc/otherpbmc_predict_nofilt_atac/peaks.bed',  header = FALSE, row.names = NULL, sep = '\t')
row.names(features) <- paste(paste(features$V1, features$V2, sep=':'), features$V3, sep='-')   #构建peak ID
features['V1']<-row.names(features)
features <- features['V1']
features
write.table(features, 'predict_otherpbmc/otherpbmc_predict_nofilt_atac/features.tsv', sep = '\t', row.names=F, col.names=F, quote = F)

count1 <- Read10X("predict_otherpbmc/otherpbmc_rna",gene.column = 1, cell.column = 1)

data1 <- Read10X("predict_otherpbmc/otherpbmc_predict__01_nofilt_atac",gene.column = 1, cell.column = 1)
colnames(data1)

# load the RNA and ATAC data

fragpath <- "10k_PBMC_Multiome_nextgem_Chromium_Controller_atac_fragments.tsv.gz"

# get gene annotations for hg38
annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotation) <- "UCSC"

# create a Seurat object containing the RNA adata
pbmc <- CreateSeuratObject(
  counts = count1,
  assay = "RNA"
)
pbmc
# create ATAC assay and add it to the object
pbmc[["ATAC"]] <- CreateChromatinAssay(
  counts = data1,
  sep = c(":", "-"),
  fragments = fragpath,
  annotation = annotation
)

DefaultAssay(pbmc) <- "ATAC"

pbmc <- NucleosomeSignal(pbmc)
pbmc <- TSSEnrichment(pbmc)

VlnPlot(
  object = pbmc,
  features = c("nCount_RNA", "nCount_ATAC", "TSS.enrichment", "nucleosome_signal"),
  ncol = 4,
  pt.size = 0
)

# filter out low quality cells
pbmc <- subset(
  x = pbmc,
  subset = nCount_ATAC < 100000 &
    nCount_RNA < 25000 &
    nCount_ATAC > 1000 &
    nCount_RNA > 1000 &
    nucleosome_signal < 2 &
    TSS.enrichment > 1
)


#Dimension reduction
DefaultAssay(pbmc) <- "RNA"
pbmc <- SCTransform(pbmc)
pbmc <- RunPCA(pbmc)

DefaultAssay(pbmc) <- "ATAC"
pbmc <- FindTopFeatures(pbmc, min.cutoff = 5)
pbmc <- RunTFIDF(pbmc)
pbmc <- RunSVD(pbmc)

library(SeuratDisk)

# load PBMC reference
reference <- LoadH5Seurat("pbmc_multimodal.h5seurat")

DefaultAssay(pbmc) <- "SCT"

# transfer cell type labels from reference to query
transfer_anchors <- FindTransferAnchors(
  reference = reference,
  query = pbmc,
  normalization.method = "SCT",
  reference.reduction = "spca",
  recompute.residuals = FALSE,
  dims = 1:50
)

reference$celltype.l2

predictions <- TransferData(
  anchorset = transfer_anchors, 
  refdata = reference$celltype.l2,
  weight.reduction = pbmc[['pca']],
  dims = 1:50
)

pbmc <- AddMetaData(
  object = pbmc,
  metadata = predictions
)

# set the cell identities to the cell type predictions
Idents(pbmc) <- "predicted.id"


write.csv(Idents(pbmc),"output.csv",)

data2 <- read.csv(
  file = "output.csv",
  row.names = 1
)

head(data2)
pbmc <- AddMetaData(object = pbmc, metadata = data2,col.name = 'predicted.id')
Idents(pbmc)<-'predicted.id'
umap<-read.csv('umap.csv',head=TRUE)
rownames(umap)<-umap$X
pbmc
umap<-umap[,-1]
head(umap)
umap
pbmc[['umap']]<-CreateDimReducObject(embeddings=data.matrix(umap),key='umap',assay=DefaultAssay(pbmc))


DimPlot(pbmc, label = TRUE,  repel = TRUE,reduction = "umap",pt.size=0.7,label.size=5,group.by='predicted.id',) + NoLegend()


# set a reasonable order for cell types to be displayed when plotting
levels(pbmc) <- c("CD4 Naive", "CD4 TCM", "CD4 CTL", "CD4 TEM", "CD4 Proliferating",
                  "CD8 Naive", "dnT",
                  "CD8 TEM", "CD8 TCM", "CD8 Proliferating", "MAIT", "NK", "NK_CD56bright",
                  "NK Proliferating", "gdT",
                  "Treg", "B naive", "B intermediate", "B memory", "Plasmablast",
                  "CD14 Mono", "CD16 Mono",
                  "cDC1", "cDC2", "pDC", "HSPC", "Eryth", "ASDC", "ILC", "Platelet")


# build a joint neighbor graph using both assays
pbmc <- FindMultiModalNeighbors(
  object = pbmc,
  reduction.list = list("pca", "lsi"), 
  dims.list = list(1:50, 2:40),
  modality.weight.name = "RNA.weight",
  verbose = TRUE
)

# build a joint UMAP visualization
pbmc <- RunUMAP(
  object = pbmc,
  nn.name = "weighted.nn",
  assay = "RNA",
  verbose = TRUE
)


DimPlot(pbmc, label = TRUE,  repel = TRUE,reduction = "umap",pt.size=0.7,label.size=5,) + NoLegend()


#Find differentially accessible peaks between clusters
DefaultAssay(pbmc) <- 'ATAC'

da_peaks1 <- FindMarkers(
  object = pbmc,
  ident.1 = "B naive",
  ident.2 = "NK",
  min.pct = 0.05,
  test.use = 'LR',
  
)
da_peaks1 <- FindMarkers(
  object = pbmc1,
  ident.1 = "Plasmablast",
  ident.2 = "CD4 Naive",
  min.pct = 0.05,
  test.use = 'LR',
)
pbmc

head(da_peaks1)
da_peaks1
plot1 <- VlnPlot(
  object = pbmc,
  features =c('chr6-167111604-167115345'),
  pt.size = 0.1,
  idents = c("CD4 Naive","B naive")
)
plot1 <- FeaturePlot(
  object = pbmc1,
  features =c('chr6-167111604-167115345'),
  pt.size = 0.1
)
plot2 <- FeaturePlot(
  object = pbmc,
  features =c('chr6-167111604-167115345'),
  pt.size = 0.1
)

plot2 | plot1

da_peaks2 <- FindMarkers(
  object = pbmc,
  ident.1 = "CD14 Mono",
  ident.2 = "B naive",
  min.pct = 0.05,
  test.use = 'LR',
)

#Jointly visualizing gene expression and DNA accessibility
DefaultAssay(pbmc) <- "ATAC"

# first compute the GC content for each peak
pbmc <- RegionStats(pbmc, genome = BSgenome.Hsapiens.UCSC.hg38)

# link peaks to genes
pbmc <- LinkPeaks(
  object = pbmc,
  peak.assay = "ATAC",
  expression.assay = "SCT",
  genes.use = c("LYZ", "MS4A1")
)

idents.plot <- c("B naive", "B intermediate", "B memory",
                 "CD14 Mono", "CD16 Mono", "CD8 TEM", "CD8 Naive")

 CoveragePlot(
  object = pbmc,
  region = "MS4A1",
  features = "MS4A1",
  expression.assay = "SCT",
  idents = idents.plot,
  extend.upstream = 500,
  extend.downstream = 10000
)

p2 <- CoveragePlot(
  object = pbmc,
  region = "LYZ",
  features = "LYZ",
  expression.assay = "SCT",
  idents = idents.plot,
  extend.upstream = 8000,
  extend.downstream = 5000
)

p2 <- CoveragePlot(
  object = pbmc,
  region = "CD14",
  features = "CD14",
  expression.assay = "SCT",
  idents = idents.plot,
  extend.upstream = 500,
  extend.downstream = 10000

)

p2<-CoveragePlot(
  object = pbmc,
  region = "CD8A",
  features = "CD8A",
  expression.assay = "SCT",
  idents = idents.plot,
  extend.upstream = 500,
  extend.downstream = 10000
  
)


patchwork::wrap_plots(p1, p2, ncol = 1)



# Get a list of motif position frequency matrices from the JASPAR database
pfm <- getMatrixSet(
  x = JASPAR2020,
  opts = list(collection = "CORE", tax_group = 'vertebrates', all_versions = FALSE)
)

DefaultAssay(pbmc) <- "ATAC"
# add motif information
pbmc <- AddMotifs(
  object = pbmc,
  genome = BSgenome.Hsapiens.UCSC.hg38,
  pfm = pfm
)

da_peaks <- FindMarkers(
  object = pbmc,
  ident.1 = "CD4 Naive",
  ident.2 = "CD4 TEM",
  only.pos = TRUE,
  test.use = 'LR',
  min.pct = 0.05,

)


# get top differentially accessible peaks
top.da.peak <- rownames(da_peaks[da_peaks$p_val < 0.005, ])

# test enrichment
enriched.motifs <- FindMotifs(
  object = pbmc,
  features = top.da.peak
)
enriched.motifs
MotifPlot(
  object = pbmc,
  motifs = head(rownames(enriched.motifs))
)
head(rownames(enriched.motifs1))
head(enriched.motifs)

MotifPlot(
  object = pbmc,
  motifs =c("MA0506.1", "MA1650.1", "MA1122.1" ,"MA1513.1", "MA0472.2" ,"MA0732.1")
)


p1<-MotifPlot(
  object = pbmc,
  motifs =c("MA1653.1", "MA0039.4", "MA1522.1", "MA1627.1")
)
p2<-MotifPlot(
  object = pbmc1,
  motifs =c("MA1653.1", "MA0039.4", "MA1522.1", "MA1627.1")
)
p1+p2





