import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

AUTOTUNE = tf.data.AUTOTUNE

def rename_close_brackets(x):
    source = x['Source']
    source = tf.strings.regex_replace(source, ']', 'X')
    source = tf.strings.regex_replace(source, r'\(', '')
    source = tf.strings.regex_replace(source, r'\)', '')
    return {'Source': source, 'Target': x['Target']}

def preprocess_dataset(file_path, tokenizer, batch_size=256):
    """Preprocess dataset."""
    print(file_path)
    sel_cols = ['Source', 'Target']
    col_defaults = [tf.string, tf.int32]
    ds = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size,
        column_defaults=col_defaults,
        select_columns=sel_cols,
        field_delim='\t',
        header=True,
        num_epochs=1
    )
    ds = ds.unbatch()
    ds = ds.map(rename_close_brackets, num_parallel_calls=AUTOTUNE)

    input_ids = []
    labels = []

    for item in ds:
        source = item['Source'].numpy().decode('utf-8')
        target = item['Target'].numpy()

        tokenized_input = tokenizer(source, add_special_tokens = False)
        input_ids.append(tokenized_input['input_ids'])

        labels.append(target)

    return dict(input_ids=input_ids, labels=labels)

