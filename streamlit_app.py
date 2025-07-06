#!/usr/bin/env python3

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap
import tempfile
import os
from pathlib import Path
import subprocess
import sys

from best_cleaned import unsupervised, supervised

st.set_page_config(
    page_title="Single-Cell RNA-seq Analysis",
    page_icon="🧬",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def create_umap_plot(file_path, clusters, params):
    try:
        adata = sc.read(file_path)
        adata.var_names = [v.split('_')[-1] for v in adata.var_names]
        adata.var_names_make_unique()
        
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=params['n_top_genes'])
        adata = adata[:, adata.var['highly_variable']].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=params['n_pcs'])
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            random_state=params['random_state']
        )
        embedding = reducer.fit_transform(adata.obsm['X_pca'][:, :min(20, adata.obsm['X_pca'].shape[1])])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='tab10', s=1, alpha=0.7)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Embedding with Cluster Assignments')
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating UMAP plot: {e}")
        return None

def supervised_with_column(train_file, test_file, output_file, label_column='cell_ontology_class'):
    if label_column == 'cell_ontology_class':
        supervised(train_file, test_file, output_file)
    else:
        import tempfile
        train_adata = sc.read(train_file)
        
        if label_column not in train_adata.obs.columns:
            raise ValueError(f"Column '{label_column}' not found in training data")
        
        train_adata.obs['cell_ontology_class'] = train_adata.obs[label_column]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp:
            train_adata.write(tmp.name)
            temp_train_path = tmp.name
        
        try:
            supervised(temp_train_path, test_file, output_file)
        finally:
            os.unlink(temp_train_path)

def main():
    st.title("Single-Cell RNA-seq Analysis")
    st.markdown("High-performance analysis pipeline for single-cell transcriptomics")
    
    st.sidebar.header("Analysis Parameters")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Unsupervised Clustering", "Supervised Classification"]
    )
    
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['h5ad', 'csv', 'tsv', 'h5', 'xlsx', 'mtx', 'loom'],
        help="Upload your single-cell data file"
    )
    
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        
        st.success(f"File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"File size: {file_size:.1f} MB")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            n_top_genes = st.slider("HVG", 500, 5000, 2000, 100)
            n_pcs = st.slider("PCA Components", 10, 100, 30, 5)
            n_neighbors = st.slider("UMAP Neighbors", 5, 50, 15, 1)
        
        with col2:
            min_dist = st.slider("UMAP Min Distance", 0.0, 1.0, 0.4, 0.05)
            random_state = st.number_input("Random Seed", value=42, min_value=0)
            
        params = {
            'n_top_genes': n_top_genes,
            'n_pcs': n_pcs,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'random_state': int(random_state)
        }
        
        if analysis_type == "Unsupervised Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5, 1)
            
            if st.button("Run Unsupervised Analysis"):
                with st.spinner("Running analysis..."):
                    output_file = file_path.replace('.h5ad', '_clusters.npy')
                    
                    try:
                        unsupervised(file_path, n_clusters, output_file)
                        
                        clusters = np.load(output_file)
                        
                        st.success("Analysis completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Clusters Found", len(np.unique(clusters)))
                            st.metric("Cells Analyzed", len(clusters))
                            
                            cluster_counts = pd.Series(clusters).value_counts().sort_index()
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cluster_counts.plot(kind='bar', ax=ax)
                            ax.set_title('Cluster Sizes')
                            ax.set_xlabel('Cluster')
                            ax.set_ylabel('Number of Cells')
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("UMAP Embedding")
                            umap_fig = create_umap_plot(file_path, clusters, params)
                            if umap_fig:
                                st.pyplot(umap_fig)
                        
                        with open(output_file, 'rb') as f:
                            st.download_button(
                                "Download Cluster Assignments",
                                f.read(),
                                file_name=f"clusters_{n_clusters}.npy",
                                mime="application/octet-stream"
                            )
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
        
        elif analysis_type == "Supervised Classification":
            st.subheader("Training Data")
            train_file = st.file_uploader(
                "Upload training data",
                type=['h5ad', 'csv', 'tsv', 'h5', 'xlsx', 'mtx', 'loom'],
                help="Upload training data with cell type labels"
            )
            
            if train_file is not None:
                train_path = save_uploaded_file(train_file)
                st.success(f"Training file uploaded: {train_file.name}")
                
                try:
                    train_adata = sc.read(train_path)
                    available_columns = list(train_adata.obs.columns)
                    
                    if available_columns:
                        st.info(f"Available columns: {', '.join(available_columns)}")
                        
                        label_column = st.selectbox(
                            "Select cell type column",
                            available_columns,
                            index=0,
                            help="Choose the column containing cell type labels"
                        )
                        
                        if label_column:
                            unique_types = train_adata.obs[label_column].unique()
                            st.write(f"**Cell types in '{label_column}':** {', '.join(map(str, unique_types[:10]))}")
                            if len(unique_types) > 10:
                                st.write(f"... and {len(unique_types) - 10} more")
                        
                        if st.button("Run Supervised Classification"):
                            with st.spinner("Running classification..."):
                                output_file = file_path.replace('.h5ad', '_predictions.npy')
                                
                                try:
                                    supervised_with_column(train_path, file_path, output_file, label_column)
                                    
                                    predictions = np.load(output_file, allow_pickle=True)
                                    
                                    st.success("Classification completed!")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Cell Types Predicted", len(np.unique(predictions)))
                                        st.metric("Cells Classified", len(predictions))
                                        
                                        type_counts = pd.Series(predictions).value_counts()
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        type_counts.plot(kind='bar', ax=ax)
                                        ax.set_title('Predicted Cell Type Distribution')
                                        ax.set_xlabel('Cell Type')
                                        ax.set_ylabel('Number of Cells')
                                        plt.xticks(rotation=45)
                                        st.pyplot(fig)
                                    
                                    with col2:
                                        fig, ax = plt.subplots(figsize=(8, 8))
                                        type_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                                        ax.set_title('Cell Type Proportions')
                                        ax.set_ylabel('')
                                        st.pyplot(fig)
                                    
                                    with open(output_file, 'rb') as f:
                                        st.download_button(
                                            "Download Predictions",
                                            f.read(),
                                            file_name="predictions.npy",
                                            mime="application/octet-stream"
                                        )
                                        
                                except Exception as e:
                                    st.error(f"Classification failed: {e}")
                                    
                except Exception as e:
                    st.error(f"Error loading training data: {e}")

if __name__ == "__main__":
    main()
