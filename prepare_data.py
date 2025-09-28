from PIL import Image
import numpy as np
import os
# from numpy.ma.core import append

def load_images(folder, label, size=(64, 64)):
    X, y = [], []  # X用于存放输入数据（图片数组）,y用来存放标签（猫= 0，狗 = 1)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    # 定义常见图片格式的扩展名
    for file in os.listdir(folder):
        # 忽略以点开头的文件（通常是隐藏文件）
        if file.startswith('.'):
            continue

        # 检查文件是否为图片
        if file.lower().endswith((image_extensions)):
            path = os.path.join(folder, file)
            try:
                img = Image.open(path).convert("RGB").resize(size)
                X.append(np.array(img) / 255.0)
                y.append(label)
            except IOError:
                print(f"警告：文件{file}无法作为图像打开，已跳过。")
    return np.array(X), np.array(y)

cat_X,cat_y = load_images(r"F:\Cat\train\cat",0)
dog_X,dog_y = load_images(r"F:\Cat\train\dog",1)
