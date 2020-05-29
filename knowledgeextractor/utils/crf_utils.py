from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
from albert import fine_tuning_utils
from albert import modeling
from albert import optimization
from albert import tokenization
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
import json
import collections

class InputExample(object):
  """A single training/test example for crf based models."""

  def __init__(self, guid, text, token_labels=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence.
      token_labels: (Optional) string. for each token in text.
    """
    self.guid = guid
    self.text = text
    self.token_labels=token_labels
    #if self.token_labels is None:
    #    self.token_labels = ["O"]* 20


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               guid=None,
               example_id=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example

def load_labels(label_list_file):
    label_list=[]
    with open(label_list_file, "r", encoding="utf-8") as f:
        for line in f:
            if line and line.strip():
                label_list.append(line.strip())
    return label_list

def convert_single_example(ex_index, example, max_seq_length,
                           processor):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokenizer=processor.tokenizer
    label_map=processor.label_map
    
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            is_real_example=False)
    
    
    
    # The convention in ALBERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    #tokens = tokenizer.tokenize(example.text)
    tokens=[]
    segment_ids = []
    label_ids=[]

    '''
    a severe bug exits using the following snippets as the original text are labeled
    character by charater ...
    we need to modify it
    it=iter(example.token_labels)
    
    for token in tokens[1:]:
        segment_ids.append(0)
        if not token.startswith(piece_prefix): # a new token rather than pieces
            label_id=label_map[next(it)]
        label_ids.append(label_id)
    '''

    '''
    a severe error also occured as pieces like cm, etc. take 2 positions chars
    offset=0
    for token in tokenizer.basic_tokenizer.tokenize(example.text):
        if example.token_labels is None:
            label_str="O"
        else:
            label_str=example.token_labels[offset]
        
        for sub_token in  tokenizer.wordpiece_tokenizer.tokenize(token):
            tokens.append(sub_token)
            label_ids.append(label_map[label_str])
            segment_ids.append(0)

        offset+=len(token.replace(piece_prefix,""))
    ''' 

    '''
    tokstr="".join(map(lambda tok: tok.replace(piece_prefix, ""),tokens[1:]))
    print("-------", tokstr)
    print("_______", example.text)
    print("******", example.token_labels)
    print(len(tokstr),len(example.token_labels),len(example.text))
    '''
    
    text=example.text
    for offset, word in enumerate(text):
        if example.token_labels is None:
            label_str="O"
        else:
            label_str=example.token_labels[offset]
        
        label=label_map[label_str]
        if word not in tokenizer.vocab:
            word="[UNK]"

        tokens.append(word )
        segment_ids.append(0)
        label_ids.append(label)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
        segment_ids=segment_ids[0:(max_seq_length-2)]
        label_ids=label_ids[0:(max_seq_length-2)]

    # inser [CLS]
    tokens.insert(0,"[CLS]")
    segment_ids.insert(0,0)
    label_ids.insert(0,label_map["O"])
    #add [SEP]
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["O"])


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    #print(len(input_ids), len(input_mask), len(segment_ids), len(label_ids))
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(label_map["O"])
    #print(len(input_ids), len(input_mask), len(segment_ids), len(label_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if example.token_labels is not None:
            tf.logging.info("label_ids: %s \n   (ids = %s)" % (" ".join(example.token_labels),
            " ".join([str(label_id) for label_id in label_ids]) ) )

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        guid=example.guid,
        example_id=ex_index,
        is_real_example=True)
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
    examples, max_seq_length, seq_processor, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example,
                                     max_seq_length, seq_processor)

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
                                drop_remainder, bsz,
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