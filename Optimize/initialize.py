import numpy as np
# 参数设置
np.random.seed(100)
p = 0.01          # 正则化参数
n_dim = 200         # 变量维度
n_nodes = 10        # 节点数量
m = 10              # 每个节点的测量维度
sparsity = 5        # 真实向量的稀疏度
max_iter = int(1e5)    # 最大迭代次数
epsilon = 1e-5      # 收敛阈值
p_values = [0.005, 0.05, 0.5, 5]  # 正则化参数列表

# 生成真实稀疏向量
x_true = np.zeros(n_dim)
nonzero_indices = np.random.choice(n_dim, sparsity, replace=False)
x_true[nonzero_indices] = np.random.normal(0, 1, 5)

# 生成测量数据
A = [np.random.randn(m, n_dim) for _ in range(n_nodes)]

noise_std = 0.1  # 噪声标准差
b = [A[i] @ x_true + noise_std * np.random.randn(m) for i in range(n_nodes)]

