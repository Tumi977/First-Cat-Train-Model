from kaggle.api.kaggle_api_extended import KaggleApi
import os

# 设置环境变量
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\ROG STRIX\.kaggle"

api = KaggleApi()
api.authenticate()

# 下载示例（如果竞赛规则已同意）
api.competition_download_files("dogs-vs-cats", path=r"F:\Cat")
