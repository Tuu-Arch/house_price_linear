import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 自动创建文件夹
if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("img"):
    os.mkdir("img")

# 解决Windows中文、负号乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 1. 构造/加载房价数据集 ======================
# 模拟真实房产数据：面积、房间数、楼层、距离地铁距离、房价(万元)
data_source = {
    "area": [62, 76, 85, 90, 103, 118, 132, 144, 158, 175, 50, 72, 96, 110, 128, 140, 165, 182],
    "room_num": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 1, 2, 3, 3, 4, 4, 5, 5],
    "floor": [6, 12, 18, 5, 11, 22, 16, 9, 15, 26, 3, 8, 14, 19, 7, 13, 21, 25],
    "subway_dist": [1.2, 2.5, 0.8, 3.1, 1.5, 0.6, 2.2, 4.0, 1.0, 0.5, 3.8, 2.8, 1.8, 0.9, 3.3, 2.0, 1.3, 0.7],
    "price": [85, 99, 123, 129, 148, 172, 188, 205, 236, 268, 68, 94, 138, 155, 182, 196, 242, 275]
}
df = pd.DataFrame(data_source)
# 导出csv保存本地
df.to_csv("data/house_data.csv", index=False, encoding="utf-8-sig")

print("="*50)
print("数据集前5行：")
print(df.head())
print("="*50)
print("数据集基础统计信息：")
print(df.describe())
print("="*50)
print("缺失值检测：")
print(df.isnull().sum())

# ====================== 2. 数据预处理 ======================
# 划分特征X、目标标签y
X = df[["area", "room_num", "floor", "subway_dist"]]
y = df["price"]

# 划分训练集80%、测试集20%，固定随机种子保证复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集样本量：{X_train.shape[0]}，测试集样本量：{X_test.shape[0]}")

# ====================== 3. 训练线性回归模型 ======================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测测试集
y_pred = lr_model.predict(X_test)

# ====================== 4. 模型评估指标 ======================
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== 模型评估结果 =====")
print(f"均方误差 MSE：{mse:.2f}")
print(f"平均绝对误差 MAE：{mae:.2f}")
print(f"拟合优度 R²：{r2:.2f}")
print("\n特征权重系数：")
for name, coef in zip(X.columns, lr_model.coef_):
    print(f"{name}：{coef:.2f}")
print(f"模型截距：{lr_model.intercept_:.2f}")

# ====================== 5. 可视化1：真实房价VS预测房价对比 ======================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="#2E86AB", s=90, label="预测房价")
# 理想拟合对角线
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2, label="完美拟合线")
plt.xlabel("真实房价（万元）")
plt.ylabel("模型预测房价（万元）")
plt.title("线性回归：真实房价 vs 预测房价")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("img/real_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()

# ====================== 6. 可视化2：面积与房价线性回归散点图 ======================
plt.figure(figsize=(7, 5))
sns.regplot(x="area", y="price", data=df, line_kws={"color": "red"}, scatter_kws={"s": 60})
plt.title("房屋面积与房价线性相关性")
plt.xlabel("房屋面积㎡")
plt.ylabel("房价 万元")
plt.grid(alpha=0.3)
plt.savefig("img/area_price_reg.png", dpi=300, bbox_inches="tight")
plt.close()

# ====================== 7. 可视化3：特征相关性热力图 ======================
plt.figure(figsize=(7, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("特征相关性热力图")
plt.savefig("img/corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# ====================== 8. 可视化4：测试集误差分布柱状图 ======================
error = y_test - y_pred
plt.figure(figsize=(7,4))
plt.bar(range(len(error)), error, color="#fc8d62")
plt.title("测试集预测误差分布")
plt.xlabel("测试样本序号")
plt.ylabel("误差（真实值-预测值）")
plt.grid(alpha=0.3)
plt.savefig("img/error_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n✅ 全部模型训练完成，4张可视化图表已保存至 img 文件夹！")