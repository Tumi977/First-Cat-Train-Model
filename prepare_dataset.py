import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
from prepare_data import cat_X, dog_X, cat_y, dog_y


# -----------------------------
# 1. 定义 SimpleNN 模型类
# -----------------------------
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # 神经元矩阵，横列大小代表传入的像素数目，纵列大小代表神经元数目
        # 随机设置初始神经元，以免所有神经元对特征的感知都一模一样
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # W2 的每一条权重连接隐藏神经元 i → 输出神经元 j。
        # W2的每一列都是对A1中这些特征向量的加权求和，根据各个特征量贡献的大小来确定是不是猫
        self.b2 = np.zeros((1, output_size))
        # 加上 偏置项（b2） 后，即使图像本身没有特别明确的猫或狗特征，网络也能有一个 初始判断
        # 这就让网络的输出更加稳定，不会因为一些小的波动（比如图像中的噪声）就偏离正确的分类。

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1  # 输入数据乘以神经元矩阵后加上一个偏置,代表每一个神经元对照片的敏感程度
        self.A1 = self.sigmoid(self.Z1)  # 激活，A1每一行每一个元素都代表着一个特征向量，比如耳朵或者尾巴，这是不同神经元对同一张照片的不同特征的分析结果
        self.Z2 = self.A1.dot(self.W2) + self.b2
        # 对每一张照片的各个神经元的评价进行加权总结，准备输出，转换成 m * 2的矩阵，W2第一列是按照猫的权重来求和的，第二列是按照狗的权重来求和的
        self.A2 = self.softmax(self.Z2)  # m * 2矩阵，0位值为猫的概率，1位为狗的概率
        return self.A2

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(m), Y])
        # Y_hat的每行是图片的预测结果，第一列是猫（对应列的0）的预测概率，第二列是狗（对应的列1）的预测概率
        # Y_hat[range(m), Y]就是取出这张图片真实结果的预测概论是多大
        loss = np.sum(log_likelihood) / m
        # 对损失的一个放大，预测的越不准确，惩罚越高（log函数的性质）（上一步只取了真实的类别的预测概率）
        return loss

    def backward(self, X, Y, learning_rate=0.001):
        m = X.shape[0]
        # One-hot 编码
        Y_onehot = np.zeros_like(self.A2)
        Y_onehot[range(m), Y] = 1

        dZ2 = self.A2 - Y_onehot  # dZ2的每一行代表一个样本，每一列代表输出概率结果与真实值的误差
        dW2 = self.A1.T.dot(dZ2) / m  # dW2 就告诉我们：这条连接需要增加还是减少权重，以及增加/减少多少
        # 这一步调节的是W2矩阵对每个神经元对特征判断的结果的权重
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # b2是对每个输出神经元的偏置输出结果的偏置，需要根据所有样本的平均误差来调整。

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.A1 * (1 - self.A1)  # A1 = F(Z1),求导即得
        dW1 = X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 更新权重
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)


# -----------------------------
# 2. 数据准备和划分
# -----------------------------
X = np.vstack([cat_X, dog_X])  # 垂直堆叠，得到总的图片数组
y = np.hstack([cat_y, dog_y])  # 水平堆叠，得到总的标签

print("合并后的 X shape:", X.shape)
print("合并后的 y shape:", y.shape)

X_flat = X.reshape(X.shape[0], -1)  # 4. 数据扁平化

X_train, X_val, y_train, y_val = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, shuffle=True
)
# test_size=0.2，验证集占20%，训练集占80%
# shuffle=True 先打乱数据，避免训练集全是猫/狗顺序影响训练
# random_state=42 固定随机种子，每次划分结果一致，便于复现

print("训练集 X_train shape:", X_train.shape)
print("训练集 y_train shape:", y_train.shape)
print("验证集 X_val shape:", X_val.shape)
print("验证集 y_val shape:", y_val.shape)


# -----------------------------
# 3. 训练和加载模型的新逻辑
# -----------------------------
def train_model(epochs=1000, learning_rate=0.001, model_path='simple_nn_model.joblib'):
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 2

    # 尝试从文件中加载模型
    if os.path.exists(model_path):
        print(f"检测到模型文件 '{model_path}'，正在加载并继续训练...")
        nn = load(model_path)
    else:
        print("未检测到模型文件，正在创建新的模型并从头开始训练...")
        nn = SimpleNN(input_size, hidden_size, output_size)

    # 开始训练循环
    for epoch in range(epochs):
        Y_hat = nn.forward(X_train)
        loss = nn.compute_loss(Y_hat, y_train)
        nn.backward(X_train, y_train, learning_rate=learning_rate)

        y_pred = nn.predict(X_val)
        acc = np.mean(y_pred == y_val)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Accuracy: {acc:.4f}")

    # 训练结束后保存模型
    dump(nn, model_path)
    print(f"\n训练完成！模型已保存到 '{model_path}'。")


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    epochs = 1000
    learning_rate = 0.01
    train_model(epochs=epochs, learning_rate=learning_rate, model_path='simple_nn_model.joblib')