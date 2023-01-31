import tensorflow as tf
import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
import os

dirname = os.path.dirname(__file__)
BERT_MODEL_NAME = "uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join(dirname, "models/uncased_L-12_H-768_A-12")
BERT_CKPT_FILE = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


def create_model(max_seq_len, num_classes, bert_ckpt_file = BERT_CKPT_FILE):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=1024, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.2)(logits)
    logits = keras.layers.Dense(units=num_classes, activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    return model