import matplotlib.pyplot as plt
from initialize import *

c = 1.0                           # 惩罚参数
# 预计算矩阵转置乘积
A_T_A = [A[i].T @ A[i] for i in range(n_nodes)]
A_T_b = [A[i].T @ b[i] for i in range(n_nodes)]

res = []
for p in p_values:
    x = [np.zeros(n_dim) for _ in range(n_nodes)]
    z = np.zeros(n_dim)
    v = [np.zeros(n_dim) for _ in range(n_nodes)]
    res_i = []
    res_i.append(z.copy())  # 存储初始值
    # ADMM迭代
    for k in range(max_iter):
        # (1) 并行更新局部变量x_i
        for i in range(n_nodes):
            M = A_T_A[i] + c * np.eye(n_dim)
            rhs = A_T_b[i] + c*z - v[i]
            x[i] = np.linalg.solve(M, rhs)
    
        # (2) 聚合更新全局变量z
        d = np.mean([x[i] + v[i]/c for i in range(n_nodes)], axis=0)
        threshold = p / (10*c)
        z_new = np.sign(d) * np.maximum(np.abs(d) - threshold, 0)
    
        # (3) 更新对偶变量v_i
        for i in range(n_nodes):
            v[i] += c * (x[i] - z_new)
    
        res_i.append(z_new.copy())
        if np.linalg.norm(z_new - z) < epsilon:
            print(f"收敛于第 {k} 次迭代")
            break
        z = z_new.copy()  # 更新z

    filename = f'ADMM_res_with_p={p}.npy'
    np.save(filename, res_i)
    res.append(res_i)

if __name__ == "__main__":
    
    distances = []
    for i in range(len(res)):
        distance_i = [np.linalg.norm(x_k - x_true) for x_k in res[i]]
        distances.append(distance_i)

    max_length = max(len(d) for d in distances)
    for i in range(4):
        distances[i] += [distances[i][-1]] * (max_length - len(distances[i]))

    for i in range(4):
        print(f"p = {p_values[i]}, Final Distance to True Value: {distances[i][-1]:.6f}")
    

    plt.figure(figsize=(12, 6))
    # 与真值的距离
    colors = ['y', 'g', 'b', 'r']
    for i in range(4):
        plt.plot(distances[i], label=f'Distance to True Value, p={p_values[i]}', color=colors[i])


    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('admm-true.png')
    plt.show()



    # 打印最后的结果（前100次迭代）
    plt.figure(figsize=(12, 6))
    # 与真值的距离
    for i in range(4):
        plt.plot(distances[i][:100], label=f'Distance to True Value, p={p_values[i]}', color=colors[i])

    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid()
    plt.savefig('admm-true_100.png')
    plt.show()
    