import glob
import pandas as pd
import re
import jieba
from pathlib import Path

# 此文件已迁移到 data_utils/data_utils.py，请从新位置导入相关方法。

def load_data(data_dir='data/'):
    data_path = Path(data_dir)
    csv_files = glob.glob(str(data_path / "*.csv"))
    dataframes = []
    for file in csv_files:
        df = None
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='gbk')
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
        if df is not None:
            dataframes.append(df)
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        raise FileNotFoundError("未找到CSV文件")

def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def preprocess_data(df, stopwords_path='stopwords.txt'):
    stopwords = stopwordslist(stopwords_path)
    df = df.dropna().sample(frac=1).reset_index(drop=True)
    df['content_clean'] = df['content'].apply(remove_punctuation)
    df['cut_content'] = df['content_clean'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    return df
