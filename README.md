# 🎯 解题思路（初步方案框架）

## 1. 静态特征处理
- One-hot / Target encoding 处理类别变量  
- 数值变量标准化 / 分桶  
- 交互特征（如 `loan_amount ÷ income`）  

## 2. 序列特征处理
深度学习禁用 → 需要用 **序列统计特征** 替代：  
- 描述统计：均值、方差、极值、趋势斜率  
- 稳定性特征：波动率、逾期次数、逾期持续时间  
- 时序模式：早期 vs 后期差异（前 N 个月均值 vs 后 N 个月均值）  
- 自相关 / 滞后特征：差分、周期性（如季度模式）  

👉 这样可以把序列转成“静态”表，方便和借款人特征拼接。  

## 3. 模型选择
- **树模型系**：LightGBM / XGBoost / CatBoost  
  - 优点：能处理类别 + 数值 + 缺失值，且对不平衡可以加 `class_weight`  
- **异常检测系**：Isolation Forest / One-Class SVM  
  - 缺点：一般效果不如监督学习强  
- **混合方案**：监督模型为主，异常检测作为 ensemble 补充  

训练集只有正常样本 → 无法用传统监督分类。

必须使用 unsupervised / semi-supervised anomaly detection：

Isolation Forest

One-Class SVM

Autoencoder（但禁止 DL，不适用这里）

LOF（Local Outlier Factor）

## 4. 类别不平衡
- SMOTE / ADASYN（合成少数类样本）  
- 欠采样多数类  
- Cost-sensitive learning：`class_weight` 调整  
- 使用 **AUC / PR-AUC** 作为指标，不用 accuracy  

## 5. 模型融合
- 单模型：LightGBM / CatBoost + CV 调优  
- 融合：Tabular (GBDT) + 异常检测 → stack/average  
- 提交：rank-based ensemble（对 anomaly score 排序再融合）  
# CS5344-FinalProject
