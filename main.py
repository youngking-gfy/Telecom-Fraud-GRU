import os
from data_utils.data_utils import load_data, preprocess_data
from model_module.model import prepare_tokenizer, prepare_sequences, prepare_labels, build_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 加载和预处理数据
df = load_data('data/')
df = preprocess_data(df, 'stopwords.txt')

# 2. 分词和标签处理
max_words = 2000
max_len = 150
embedding_dim = 200
num_classes = df['label'].nunique()
tokenizer = prepare_tokenizer(df['cut_content'].values, max_words)
X = prepare_sequences(tokenizer, df['cut_content'].values, max_len)
Y = prepare_labels(df['label'])

# 3. 划分数据集
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# 4. 构建和训练模型
model = build_model(max_words, embedding_dim, num_classes)
early_stop = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, mode='auto')
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=20,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 5. 评估与可视化
model.summary()
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()

score = model.evaluate(X_test, Y_test)
print(f"Test loss: {score[0]}, Test accuracy: {score[1]}")

# 6. 保存模型
model.save('model_GRU.h5')
