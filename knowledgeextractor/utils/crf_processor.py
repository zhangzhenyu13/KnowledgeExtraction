from albert import tokenization
from knowledgeextractor.utils.crf_utils import convert_single_example, InputExample, InputFeatures
from knowledgeextractor import KGEConfig
import tensorflow as tf
import numpy as np
from tensorflow.contrib import data as contrib_data
import collections
from knowledgeextractor.utils.processor import TextProcessor
import os
import json
from albert.classifier_utils import DataProcessor


class TaskTrainProcessor(DataProcessor):
  """Processor for the NER data set (CRF version)."""
  def __init__(self, use_spm=False, do_lower_case=False):
    super(TaskTrainProcessor, self).__init__(use_spm, do_lower_case)

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "CRF", "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "CRF", "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "CRF", "test.json")), "test")

  def get_labels(self,data_dir):
    """See base class."""
    labels_list=[]
    with open( os.path.join(data_dir, "CRF", "labels"), "r" ) as f:
        for line in f:
            labels_list.append(self.process_text(line).strip())
    return labels_list
    
  def _read_json(self, file_name):
      lines=[]
      with open(file_name,"r", encoding="utf-8") as f:
          for line in f:
              lines.append(line)
      return lines
      
  def _create_examples(self, lines, set_type):
    """Creates examples from json lines."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      data=json.loads(line)
      text=self.process_text(data["text"])
      
      if set_type == "test":
        guid = line[0]
        token_labels= None
      else:
        token_labels= data["token_labels"]
      examples.append(
          InputExample(guid=guid, text=text, token_labels=token_labels))
    return examples

class SquenceLabelingTextProcessor(TextProcessor):
    def __init__(self, config_file):
        self.config=KGEConfig(config_file=config_file)
        config=KGEConfig(config_file=config_file)
        self.tokenizer=tokenization.FullTokenizer( config.vocab )
        self.max_sequence_length=config.max_sequence_length
        self.doc_stride=config.doc_stride
        self.label_list=list(config.label_list)
        self.label_map={label:index for index, label in enumerate(self.label_list)}
        self.inv_label_map={v:k for k, v in self.label_map.items()}
        self.piece_prefix="##"
    
    def recoverText(self, tokens,labels):
        # tokens are [CLS] tok1, tok2, ... [SEP] [PAD] [PAD] ...
        # extract the part marked as tokN
        words=[]
        word=""
        labels2=[]
        it= iter(labels)
        next(it) # drop first [CLS]
        for tok in tokens[1:]:
            if tok=="[SEP]":
                break
            label=next(it)
            if tok.startswith(self.piece_prefix):
                word+=tok[2:]
            else: #begin of a word
                if word:
                    words.append(word)
                labels2.append(label) # push label for current word
                word=tok
        if word:
            words.append(word)
        return words, labels2

    def recover_token_tags(self, label_ids, input_ids):
        labels=[]
        print("inv label map", self.inv_label_map)
        for label_id in label_ids:
            labels.append(self.inv_label_map[label_id])
        
        tokens=self.tokenizer.convert_ids_to_tokens(input_ids)
        words, labels2= self.recoverText(tokens,labels)
        return words, labels2

    def processText(self, query_data):
        '''
        {
            "guid":id,
            "text":....,

        }
        '''
        guid=query_data["guid"]
        text=query_data["text"]
        example=InputExample(guid=guid, text=text)
        ex_index=0
        label_list=self.label_list
        max_seq_length=self.max_sequence_length
        tokenizer=self.tokenizer
        feature=convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer)
        '''def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        
        return features
        '''
        return feature



def data_based_input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, task_name):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] =  create_int_feature(feature.label_ids)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.int64

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length * multiple], labeltype),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

#--------test---------------
def test_processor():
    text="what is elephant talker?"
    labels=["O","O", "O", "B", "B", "B", "B","I", "I", "O", "O"] #after tokenized
    processor=SquenceLabelingTextProcessor("/home/zhangzy/KnowledgeExtraction/config/crf_processor.json")
    tokens=processor.tokenizer.tokenize(text)
    tokens=["[CLS]"]+tokens+["[SEP]"]
    print(tokens)
    ids=processor.tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    tokens=processor.tokenizer.convert_ids_to_tokens(ids)
    print(tokens)

    words, labels= processor.recoverText(tokens, labels)
    print(labels)
    print(words)
    exit(0)

if __name__ == "__main__":
    test_processor()