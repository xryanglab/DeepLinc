# DeepLinc
De novo reconstruction of cell interaction landscapes from single-cell spatial transcriptome data.

# VERSION

1.0.0

# Authors

Runze Li and Xuerui Yang

# REQUIREMENTS

* Python --- 3.5.2 (Anaconda3-4.2.0)
* Numpy --- 1.16.4
* Scipy --- 1.2.1
* Pandas --- 0.20.3
* Matplotlib --- 3.0.2
* Seaborn --- 0.8.1
* Networkx --- 2.1
* Scikit-learn --- 0.21.2
* Tensorflow --- 1.4.0

# USAGE

python DeepLinc.py -e ./dataset/seqFISH/counts.csv -a ./dataset/seqFISH/adj.csv -c ./dataset/seqFISH/coord.csv -r ./dataset/seqFISH/cell_type_1.csv

The final output reports the AUPRC performance, the reconstructed cell adjacency matrix, the over- or under-representation of interaction between cell groups, the latent feature for each cell and the saved model.

# LICENSE

DeepLinc is licensed under the MIT license

