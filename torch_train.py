import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import cat_X, dog_X, cat_y, dog_y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)  # 输出: cuda 或 cpu
# 1. 数据准备
X = np.vstack([cat_X, dog_X])  # 合并猫狗图片
y = np.hstack([cat_y, dog_y])  # 合并标签

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 数据扁平化
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_train_flat = X_train_flat / 255.0
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_val_flat = X_val_flat / 255.0

# 2. PyTorch 数据转换
X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_flat, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 先前准备的数据集转换为tensor形式的张量
# tensor张量其实就是一个高维矩阵，只不过维度更加灵活

# 3. 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.sigmoid(self.fc1(X))
        X = self.softmax(self.fc2(X))
        return X

# 损失函数与优化器
model = SimpleNN(input_size=X_train_flat.shape[1], hidden_size=64, output_size=2)
loss_fn = nn.CrossEntropyLoss()  # 用于分类的交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器

# 如果有已保存的模型参数，加载它们
try:
    model.load_state_dict(torch.load('model.pth'))
    print("模型参数加载成功！")
except FileNotFoundError:
    print("没有找到已保存的模型参数，开始新训练。")

# 训练模型
model.train()  # 训练模式

epochs = 1000
for epoch in range(epochs):
    # 前向传播
    Y_hat = model(X_train_tensor)
    loss = loss_fn(Y_hat, y_train_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证准确率
    model.eval()  # 切换为评估模式
    with torch.no_grad():
        y_pred = model(X_val_tensor)
        acc = (y_pred.argmax(dim=1) == y_val_tensor).float().mean()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Val Accuracy: {acc.item():.4f}")

    # 每个epoch结束后保存模型参数
    torch.save(model.state_dict(), 'model.pth')
    print("模型参数已保存！")