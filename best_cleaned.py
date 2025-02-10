#!/usr/bin/env python3
import numpy as np
import umap
import faiss
import scanpy as sc
import argparse
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans 
from scipy.sparse import issparse

warnings.simplefilter(action='ignore', category=FutureWarning)

def to_dense(X):
    return X.toarray() if issparse(X) else X

def preprocess(
    adata,
    filter_hvg=True,
    n_top_genes=2000,
    do_scale=True,
    do_pca=True,
    n_pcs=30
): 
    adata.var_names = [v.split('_')[-1] for v in adata.var_names]
    adata.var_names_make_unique()
    
    sc.pp.filter_genes(adata, min_cells=3)

    if filter_hvg:
        sc.pp.highly_variable_genes(
            adata,
            flavor='cell_ranger',   
            n_top_genes=n_top_genes
        ) 

        adata = adata[:, adata.var['highly_variable']].copy()
    #library size normalization 10,000 , log transform, and scaling
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if do_scale:
        sc.pp.scale(adata, max_value=10)

    if do_pca:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)

    return adata


def get_embedding(
    adata,
    use_pca=True,
    umap_dim=10,
    n_neighbors=15,
    min_dist=0.4,
    random_state=42
):

    if use_pca:
        X_for_umap = adata.obsm['X_pca'][:, :min(umap_dim*2, adata.obsm['X_pca'].shape[1])]
    else:
        X_for_umap = to_dense(adata.X)

    reducer = umap.UMAP(
        n_components=umap_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    emb = reducer.fit_transform(X_for_umap)
    return emb

def unsupervised(data_file, k, out_file):
    adata = sc.read(data_file)
    adata = preprocess(adata, filter_hvg=True, n_top_genes=2000, do_scale=True, do_pca=True, n_pcs=30)
    embedding = get_embedding(
        adata,
        use_pca=True,
        umap_dim=10,       
        n_neighbors=15,    
        min_dist=0.4       
    ).astype(np.float32)

    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
    clusters = kmeans.fit_predict(embedding)

    np.save(out_file, clusters.astype(np.int64), allow_pickle=False)

def supervised(train_file, test_file, out_file):
    train = sc.read(train_file)
    test = sc.read(test_file)

    train.var_names = [v.split('_')[-1] for v in train.var_names]
    train.var_names_make_unique()
    test.var_names = [v.split('_')[-1] for v in test.var_names]
    test.var_names_make_unique()
    common_genes = np.intersect1d(train.var_names, test.var_names)
    train = train[:, common_genes].copy()
    test = test[:, common_genes].copy()

    combined = train.concatenate(test, join='outer', batch_categories=['train','test'])
    combined = preprocess(combined, filter_hvg=True, n_top_genes=2000, do_scale=True, do_pca=True, n_pcs=30)
    combined_emb = get_embedding(
        combined,
        use_pca=True,
        umap_dim=10,
        n_neighbors=15,
        min_dist=0.4
    )

    n_train = train.shape[0]
    train_emb = combined_emb[:n_train]
    test_emb = combined_emb[n_train:]

    rf = RandomForestClassifier(
        n_estimators=200,  
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(train_emb, train.obs['cell_ontology_class'].values)
    preds = rf.predict(test_emb)
    preds = preds.astype(np.str_)
    np.save(out_file, preds, allow_pickle=False)

def main():
    parser = argparse.ArgumentParser(description='Improved single-cell clustering/labeling pipeline using HVG, PCA, and UMAP.')
    parser.add_argument('-d','--data',help='Input anndata file (or test file for supervised)',required=True)
    parser.add_argument('-k',type=int,help='Number of clusters (for unsupervised clustering)')
    parser.add_argument('-t','--train_data',help='Training anndata file (for supervised labeling)')
    parser.add_argument('-o','--output_file',help='Output file (npy)',required=True)
    args = parser.parse_args()

    if args.train_data:
        # Supervised
        supervised(args.train_data, args.data, args.output_file)
    else:
        # Unsupervised
        unsupervised(args.data, args.k, args.output_file)

if __name__ == '__main__':
    main()