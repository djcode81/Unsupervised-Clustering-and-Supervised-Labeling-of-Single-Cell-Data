# **Unsupervised Clustering and Supervised Labeling of Single-Cell Data**
This project implements both **unsupervised clustering** and **supervised labeling** of single-cell RNA sequencing data. It leverages **highly variable gene selection (HVG), PCA for dimensionality reduction, UMAP embeddings, K-Means clustering, and Random Forest classification** to analyze single-cell transcriptomic datasets.

## **Components**
- **Unsupervised clustering** using PCA, UMAP, and K-Means.
- **Supervised classification** using Random Forest trained on labeled data.
- **Efficient preprocessing pipeline** that normalizes, log-transforms, and selects highly variable genes.
- **Scalable and modular implementation** using Python and Scanpy.

## **Files**
- `best_cleaned.py` - Main script for both clustering and classification.
- `data_exp.py` - Exploratory data analysis script.
- `clusters.npy` - Output file containing the predicted cluster labels.
- `requirements.txt` - List of required Python dependencies.

## **Usage**

python3 best_cleaned.py -t mouse_spatial_brain_section0.h5ad -d mouse_spatial_brain_section1_modified.h5ad -o clusters.npy # supervised labeling

The pipeline was evaluated using Adjusted Mutual Information (AMI) scores and ended scoring the highest in class 😎


Acknowledgments : 
This project was completed as part of the MSCBIO2066 coursework at University of Pittsburgh.








https://bits.csb.pitt.edu/mscbio2066/assign3/data/
