# DeepLinc: deep-learning framework for landscapes of interacting cells
DeepLinc is a tool for de novo reconstruction of cell interaction landscapes from single-cell spatial transcriptome data. DeepLinc provides utilities to (1) learn from the incomplete and noisy predefined set of cell-cell interactions (2) **remove false positive local interactions** and reconstruct existing interactions (3) restore and regenerate a more unbiased and complete landscape of cell-cell interactions **including both proximal and distal interactions** (4) evaluate **the over- or under-representation of interactions between cell types** (5) extract **the latent features** related to cell interactions (6) identify **multicellular domains** informing organizational tissue structures.

## Version

1.0.0

## Authors

Runze Li and Xuerui Yang

## Getting Started

### Dependencies and requirements

DeepLinc depends on the following packages: numpy, pandas, scipy, matplotlib, seaborn, networkx, scikit-learn, umap-learn, tensorflow. See dependency versions in `requirements.txt`. The package has been tested on Anaconda3-4.2.0 and is platform independent (tested on Windows and Linux) and should work in any valid python environment. To speed up the training process, DeepLinc relies on Graphic Processing Unit (GPU). If no GPU device is available, the CPU will be used for model training. No special hardware is required besides these. Installation of the dependencies may take several minutes.

```
pip install --requirement requirements.txt
```

### Usage
Assume we have (1) a CSV-formatted raw count matrix ``counts.csv`` with cells in rows and genes in columns (2) a coordinate file ``coord.csv`` including X and Y columns (3) an adjacent matrix in ``adj.csv`` as a predefined local interaction map (4) a cell type annotation file ``cell_type.csv`` including columns Cell_ID, Cell_class_id and Cell_class_name. The cell type information is not essential for reconstructing cell interaction landscapes. We will provide a preprocessing module to help users transform the general coordinate information from single-cell spatial transcriptome data into the adjacency matrix.

You can run a demo from the command line:

``python DeepLinc.py -e ./dataset/seqFISH/counts.csv -a ./dataset/seqFISH/adj.csv -c ./dataset/seqFISH/coord.csv -r ./dataset/seqFISH/cell_type_1.csv``


### Results

The final output reports the AUPRC performance, the reconstructed cell adjacency matrix, the over- or under-representation of interaction between cell groups, the latent feature for each cell and the saved model.

## License

DeepLinc is licensed under the MIT license

