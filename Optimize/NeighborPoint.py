import numpy as np
import matplotlib.pyplot as plt
from initialize import *

res_admm = []
res = []  # 存储迭代解
l = len(p_values)

# 估计 Lipschitz 常数 L = sum_i ||A_i^T A_i||_2
L_est = sum(np.linalg.norm(A_i.T @ A_i, 2) for A_i in A)
alpha = 1.0 / (L_est + 1e-8)  # 步长

for p in p_values:
    # 初始化变量
    x = np.zeros(n_dim)
    res_admm_i = np.load(f'ADMM_res_with_p={p}.npy', allow_pickle=True)
    res_i = []
    res_i.append(x.copy())  # 存储初始值
    for k in range(max_iter):
        # (1) 梯度下降步
        grad = np.zeros_like(x)
        for A_i, b_i in zip(A, b):
            grad += A_i.T @ (A_i @ x - b_i)
        x_hat = x - alpha * grad

        # (2) 软门限
        x_new = np.sign(x_hat) * np.maximum(np.abs(x_hat) - p * alpha, 0)

        # 存储并检查收敛
        res_i.append(x_new.copy())
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"收敛于第 {k} 次迭代")
            break
        x = x_new
    res_admm.append(res_admm_i)
    res.append(res_i)

if __name__ == "__main__":
    
    # 收敛分析：与真值距离
    distance_true = []
    distance_admm = []
    for p in range(len(p_values)):
        distance_true_i = [np.linalg.norm(x_k - x_true) for x_k in res[p]]
        distance_admm_i = [np.linalg.norm(x_k - x_admm) for x_k, x_admm in zip(res[p], res_admm[p])]

        # res_admm 的长度和 res 的长度不同，补齐  
        if len(res[p]) > len(res_admm[p]):
            distance_admm_i.extend([np.linalg.norm(res_admm[p][-1] - res[p][j]) for j in range(len(res_admm[p]), len(res[p]))])
        else:
            distance_admm_i.extend([np.linalg.norm(res[p][-1] - res_admm[p][j]) for j in range(len(res[p]), len(res_admm[p]))])

        distance_true.append(distance_true_i)
        distance_admm.append(distance_admm_i)

    max_length = max(max(len(d) for d in distance_true), max(len(d) for d in distance_admm))
    for i in range(4):
        distance_true[i] += [distance_true[i][-1]] * (max_length - len(distance_true[i]))
        distance_admm[i] += [distance_admm[i][-1]] * (max_length - len(distance_admm[i]))


    for i in range(l):
        print(f"p = {p_values[i]}, Final Distance to True Value: {distance_true[i][-1]:.6f}")
        print(f"p = {p_values[i]}, Final Distance to ADMM Value: {distance_admm[i][-1]:.6f}")    

    plt.figure(figsize=(12, 6))

    color_map = ['y', 'g', 'b', 'r']

    for i in range(l):
        # 绘制与真值的距离曲线
        plt.plot(distance_true[i], label=f'Distance to True Value, p={p_values[i]}', color=color_map[i])


    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('ista-true.png')
    plt.show()


    for i in range(l):
        # 绘制与 ADMM 的距离曲线
        plt.plot(distance_admm[i], label=f'Distance to ADMM Value, p={p_values[i]}', color=color_map[i])


    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('ista-admm.png')
    plt.show()


        