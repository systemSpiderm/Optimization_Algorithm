import numpy as np
import matplotlib.pyplot as plt
from initialize import *

res_admm = []
res = []
l = len(p_values)

def clip_gradient(grad, clip_value=1.0):
    grad_norm = np.linalg.norm(grad)
    if grad_norm > clip_value:
        grad = grad / grad_norm * clip_value
    return grad


for p in p_values:
    # 次梯度下降 主循环
    x = np.zeros(n_dim)
    res_admm_i = np.load(f'ADMM_res_with_p={p}.npy', allow_pickle=True)
    res_i = []
    res_i.append(x.copy())  # 存储初始值

    for k in range(max_iter):
        alpha_k = 1 / np.sqrt(k + 1)

        # (1) 计算可微部分的梯度 ∇S(x)
        grad_S = np.zeros_like(x)
        for A_i, b_i in zip(A, b):
            grad_S += A_i.T @ (A_i @ x - b_i)

        grad_S = clip_gradient(grad_S, clip_value=10.0)  # 梯度裁剪

        # (2) 计算非光滑部分的次梯度 ∂r(x)
        subgrad_r = np.zeros_like(x)
        for i in range(n_dim):
            if x[i] > 0:
                subgrad_r[i] = p
            elif x[i] < 0:
                subgrad_r[i] = -p
            else:
                # x[i] == 0 时随机选取次梯度
                subgrad_r[i] = np.random.uniform(-p, p)
        # (3) 更新 x
        x_new = x - alpha_k * (grad_S + subgrad_r)

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

    for i in range(l-1):
        # 绘制与真值的距离曲线
        plt.plot(distance_true[i][:1000], label=f'Distance to True Value, p={p_values[i]}', color=color_map[i])


    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('subgrad-true.png')
    plt.show()


    for i in range(l-1):
        # 绘制与 ADMM 的距离曲线
        plt.plot(distance_admm[i][:1000], label=f'Distance to ADMM Value, p={p_values[i]}', color=color_map[i])


    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('subgrad-admm.png')
    plt.show()