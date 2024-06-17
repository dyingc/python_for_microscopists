import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
n_samples = 1000
n_features = 1536
centers = 16
reduced_dim = 3
cluster_std = (np.random.rand(centers) + 0.5) * 10 - 4 # 生成随机的标准差: [1, 11)

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)

# 使用 UMAP 进行降维
n_neighbors = 15  # 控制邻域大小
umap_model = umap.UMAP(n_neighbors=n_neighbors, random_state=42, n_components=reduced_dim)
X_umap = umap_model.fit_transform(X) # X_umap.shape = (n_samples, reduced_dim)

# 绘制降维结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=y, cmap='viridis', s=50)
plt.title(f"UMAP with n_neighbors={n_neighbors}")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_zlabel("UMAP3")

# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Cluster Label')

plt.show()