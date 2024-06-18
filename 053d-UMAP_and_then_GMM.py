import umap
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# 生成高维数据
n_samples = 1000
n_features = 1536  # 高维数据
centers = 8
center_pos = np.random.normal()
cluster_std = (np.random.rand(centers) + 0.1) * 5 # 生成随机的标准差: [0.5, 11)

X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)

# 使用UMAP降维到低维
n_neighbors = 15
n_components = 64
umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
X_umap = umap_model.fit_transform(X)

# bic准则选择最优的GMM模型
def find_best_params_by_bic(data, n_components_range=None, grace_period=0):
    if n_components_range is None:
        n_components_range = range(1, len(data))
    global_lowest_bic = np.infty # The lowest bic across all covariance types
    best_n_component = n_components_range[0]
    best_type = "spherical"
    bics = []
    cv_types = ['tied', 'diag', 'full', 'spherical']
    for cv_type in cv_types:
        local_lowest_bic = np.infty # The lowest bic for this covariance type
        increasing_bic_count = 0  # count the number of times BIC increases consecutively
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
            try:
                gmm.fit(data)
            except ValueError as ve:
                print(f"\n\nError fitting model with n_components={n_components} and type={cv_type}: {ve}\n\n")
                break
            bic = gmm.bic(data)
            bics.append(bic)
            if bic < local_lowest_bic:
                local_lowest_bic = bic
                if local_lowest_bic < global_lowest_bic:
                    global_lowest_bic = local_lowest_bic
                    best_n_component = n_components
                    best_type = cv_type
                    best_gmm = gmm
                increasing_bic_count = 0  # reset the count if BIC decreases
                print(f"Found better model with BIC={local_lowest_bic} and n_components={n_components} and type={cv_type}")
            else:
                print(f"Wrose BIC found: BIC={bic} and n_components={n_components} and type={cv_type}")
                increasing_bic_count += 1  # increment the count if BIC increases
            # stop if BIC increases more than the grace period
            if increasing_bic_count > grace_period:
                print("\n\n")
                break # test the next covariance type
    print(f"Best model has BIC={global_lowest_bic} and n_components={best_n_component} and type={best_type}")
    return best_gmm, best_n_component, best_type, bics

# 使用GMM进行聚类
best_gmm, best_n_component, best_type, bics = find_best_params_by_bic(X_umap)
gmm = GaussianMixture(n_components=best_n_component, covariance_type=best_type, random_state=42)
gmm.fit(X_umap)
labels = gmm.predict(X_umap)
# gmm = best_gmm

# 为了可视化，将数据再降维到2维
umap_2d = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
X_umap_2d = umap_2d.fit_transform(X_umap)

# 绘制降维和聚类结果
plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], c=labels, cmap='viridis', s=50)
plt.title(f"UMAP to 64D + GMM Clustering (n_neighbors={n_neighbors})")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='Cluster Label')
plt.show()