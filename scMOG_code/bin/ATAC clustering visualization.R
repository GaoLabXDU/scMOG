#ATAC Downstream Analysis

library(SeuratDisk)
library(patchwork)
library(Seurat)
library(Signac)
library(GenomeInfoDb)
library(EnsDb.Hsapiens.v75)
library(patchwork)
library(RColorBrewer)


data1 <- read.csv(
  file = "predicted_atac_nofilted_01.csv",
  header = TRUE,
  row.names = 1
)
head(data1)

data2 <- read.csv(
  file = "pbmc_rna_leiden_01.csv",
  header=FALSE,
  row.names = 1
)




chrom_assay <- CreateChromatinAssay(counts = data1,sep = c(":", "-"))

pbmc <- CreateSeuratObject(counts = chrom_assay,assay = "peaks")
rownames(data2)<-colnames(pbmc)
head(data2)
pbmc <- AddMetaData(object = pbmc, metadata = data2,col.name = 'letter.idents')
granges(pbmc)
head(pbmc)


pbmc <- RunTFIDF(pbmc)
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q25')
pbmc <- RunSVD(pbmc)
DepthCor(pbmc)
pbmc <- RunUMAP(object = pbmc, reduction = 'lsi', dims = 2:30)
pbmc <- FindNeighbors(object = pbmc, reduction = 'lsi', dims = 2:30)
pbmc <- FindClusters(object = pbmc, verbose = FALSE, algorithm = 3)
plot1=DimPlot(object = pbmc, label = TRUE,group.by='letter.idents') + NoLegend()
plot1_1=DimPlot(object = pbmc, label = TRUE,group.by='letter.idents',pt.size=0.7,label.size=5,) + NoLegend()
plot1_1


gene.activities <- GeneActivity(pbmc)
# add the gene activity matrix to the Seurat object as a new assay and normalize it
pbmc[['RNA']] <- CreateAssayObject(counts = gene.activities)
pbmc <- NormalizeData(
  object = pbmc,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(pbmc$nCount_RNA)
)

DefaultAssay(pbmc) <- 'RNA'

FeaturePlot(
  object = pbmc,
  features = c('MS4A1', 'CD3D', 'LEF1', 'NKG7', 'TREM1', 'LYZ'),
  pt.size = 0.1,
  max.cutoff = 'q95',
  ncol = 3
)










