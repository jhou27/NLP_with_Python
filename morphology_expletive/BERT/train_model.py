import build_dataset
import preprocess
import build_model
import keras
import os
from bert.tokenization.bert_tokenization import FullTokenizer

dirname = os.path.dirname(__file__)
DATASET_PATH = os.path.join(dirname, "/coding/data/unique_tweets_7k.csv")
VOCAB_PATH = os.path.join(dirname, "models/uncased_L-12_H-768_A-12/vocab.txt")
MAX_SEQ_LEN = 40
tokenizer = FullTokenizer(vocab_file=VOCAB_PATH)

train_val, test = build_dataset.prepare_train_test_from_file(DATASET_PATH)
data = preprocess.SentimentAnalysisData(train_val, test, tokenizer, max_seq_len=MAX_SEQ_LEN)
model = build_model.create_model(data.max_seq_len, 5)
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

history = model.fit(
    x=data.train_x,
    y=data.train_y,
    validation_split=0.2,
    batch_size=32,
    shuffle=True,
    epochs=12,
    verbose=1
)

_, test_acc = model.evaluate(data.test_x, data.test_y)
_, train_acc = model.evaluate(data.train_x, data.train_y)
print("Test Accuracy:" + str(test_acc))