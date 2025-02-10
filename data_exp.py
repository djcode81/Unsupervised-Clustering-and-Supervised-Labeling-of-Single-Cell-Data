import scanpy as sc

# Load datasets
adata1 = sc.read_h5ad("/Users/mouse_spatial_brain_section1_modified.h5ad")
adata0 = sc.read_h5ad("/Users/mouse_spatial_brain_section0.h5ad")

# Print basic info
print(adata1)
print(adata0)
