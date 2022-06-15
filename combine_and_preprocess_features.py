import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
np.seterr(divide='ignore', invalid='ignore')

print("\nLoading data ...")
atom_features = pd.read_feather("data/atom_features_PCA.feather")
bond_features = pd.read_feather("data/bond_features_PCA.feather")
global_features = pd.read_feather("data/molecule_descriptors.feather")

n_components = 50
print(f"Getting top {n_components} local features ...")
local_features = pd.concat(objs=[atom_features, bond_features], axis=1)
local_features = local_features.to_numpy(dtype=np.float32)
top50_local_features = PCA(n_components=n_components).fit_transform(local_features)

print("Cleaning global features ...")
global_features.fillna(0., inplace=True)
global_features = global_features.iloc[:,1:].to_numpy(dtype=np.float32)

print("Z-score scaling global features ...")
mean_global_features = np.mean(global_features, axis=0)
std_global_features = np.std(global_features, axis=0)
zscore_scaled_global_features = (global_features - mean_global_features) / std_global_features

print(f"Combining local and global features to an array of size {global_features.shape[1] + n_components} ...")
all_features = np.concatenate([global_features, top50_local_features], axis=1)

# filter out zero-variance columns
variance_matrix = np.var(all_features, axis=0)
indices_with_zero_variance = np.argwhere(variance_matrix == 0)
print(f"Dropping {indices_with_zero_variance.shape[0]} colums with 0 variance ...")
filtered_features = np.delete(all_features, indices_with_zero_variance, axis=1)

print("Saving final features ...")
np.save("data/final_features.npy", filtered_features)
print("Done! \U0001F389\n")