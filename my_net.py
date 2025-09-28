import numpy as np

# ====== 1. 造一批模拟数据 ======
# 生成两类点：一类(猫)围绕(0,0)，另一类(非猫)围绕(2,2)
np.random.seed(42)
N = 200  # 每类样本数
cat_data = np.random.randn(N, 2) * 0.5 + np.array([0, 0])
not_cat_data = np.random.randn(N, 2) * 0.5 + np.array([2, 2])
X = np.vstack([cat_data, not_cat_data])       # shape: (400, 2)
y = np.hstack([np.zeros(N), np.ones(N)])      # 0=猫, 1=非猫
y = y.reshape(-1, 1)                          # 列向量

# 数据归一化
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ====== 2. 定义网络结构 ======
# 输入层(2维) → 隐藏层(4个神经元，ReLU) → 输出层(1个神经元，Sigmoid)
def init_params():
    W1 = np.random.randn(2, 4) * 0.01  # 权重
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# 激活函数
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z))

# ====== 3. 前向传播 ======
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1     # (400,4)
    a1 = relu(z1)
    z2 = a1 @ W2 + b2    # (400,1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# ====== 4. 损失函数 ======
def compute_loss(y, a2):
    # 二分类交叉熵
    m = y.shape[0]
    eps = 1e-8
    return -np.mean(y*np.log(a2+eps) + (1-y)*np.log(1-a2+eps))

# ====== 5. 反向传播 ======
def backward(X, y, z1, a1, a2, W2):
    m = X.shape[0]
    dz2 = a2 - y                       # 输出层误差
    dW2 = (a1.T @ dz2) / m
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * relu_deriv(z1) # 隐藏层误差
    dW1 = (X.T @ dz1) / m
    db1 = np.mean(dz1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

# ====== 6. 参数更新 ======
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# ====== 7. 训练循环 ======
W1, b1, W2, b2 = init_params()
lr = 0.1
for epoch in range(2000):
    z1,a1,z2,a2 = forward(X, W1, b1, W2, b2)
    loss = compute_loss(y, a2)
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, a2, W2)
    W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,lr)

    if epoch % 200 == 0:
        preds = (a2 > 0.5).astype(int)
        acc = (preds == y).mean()
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

# ====== 8. 测试预测 ======
test = np.array([[0,0],[2,2],[1,1]])
test = (test - X.mean(axis=0)) / X.std(axis=0)
_,_,_,test_out = forward(test, W1,b1,W2,b2)
print("测试结果(越接近1越像'非猫'):\n", test_out)
