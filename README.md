# 文本分类 GRU 模型项目

本项目实现了基于 GRU 的中文文本分类，包含数据预处理、分词、模型训练与评估等完整流程。

## 目录结构

- data_utils.py：数据加载与预处理模块
- model.py：模型构建与训练相关模块
- main.py：主运行脚本
- requirements.txt：依赖包列表
- stopwords.txt：中文停用词表
- data/：存放原始 CSV 数据文件

## 快速开始

1. 下载数据集
   [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5)
   下载后放在 data/ 目录下
2. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 运行主程序

   ```bash
   python main.py
   ```

4. 运行 my1.ipynb
   ```bash
   jupyter notebook my1.ipynb
   ```
   可以直接查看结果

## 依赖环境

- Python 3.6+
- TensorFlow/Keras
- pandas、numpy、jieba、matplotlib、scikit-learn

## 数据说明

- data/ 目录下应包含带有 content 和 label 字段的 CSV 文件。
- stopwords.txt 为中文停用词表。

## 结果

- 训练完成后会生成 model_GRU.h5 模型文件。
