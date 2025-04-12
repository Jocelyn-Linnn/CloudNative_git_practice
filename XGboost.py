#pip install xgboost

import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# 讀取資料
X_train = pd.read_csv('data/X_train.csv', header=0, usecols=lambda x: x != 0).values
y_train = pd.read_csv('data/y_train.csv', header=0, usecols=[1]).values
X_test = pd.read_csv('data/X_test.csv', header=0, usecols=lambda x: x != 0).values

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 標準化目標值（房價）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# 使用訓練數據集和測試數據集進行模型訓練和預測
X_train_final = X_train_scaled
y_train_final = y_train_scaled.flatten()  # 必須確保 y_train 是一維的


# 使用 XGBoost 訓練模型
dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
params = {
    'objective': 'reg:squarederror',  # 用於回歸問題
    'max_depth': 20,                   # 樹的最大深度
    'eta': 0.01,                       # 學習率
    'subsample': 0.9,                 # 隨機抽樣比例
    'colsample_bytree': 0.9,          # 每棵樹使用的特徵比例
    'eval_metric': 'mae',            # 評估指標
    'verbose': 0
}

# 訓練 XGBoost 模型
num_round = 500
bst = xgb.train(params, dtrain, num_round)


# 預測
dtest = xgb.DMatrix(X_test_scaled)
y_pred_scaled = bst.predict(dtest)

# 將預測值轉換回原始房價範圍
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred = np.round(y_pred).astype(int)


# 儲存預測結果
output = pd.DataFrame({'Index': range(0, len(y_pred)), 'Price': y_pred.flatten()})
output.to_csv('predictions_xgboost.csv', index=False)

print("預測完成並保存為 predictions_xgboost.csv")

# 計算模型性能
y_train_pred = bst.predict(dtrain)
mae_train = mean_absolute_error(y_train_final, y_train_pred)

print(f"Train MAE: {mae_train:.4f}")
