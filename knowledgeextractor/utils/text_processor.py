from albert.squad_utils import convert_examples_to_features, SquadExample
from albert import tokenization
from knowledgeextractor.utils.crf_utils import convert_single_example, InputExample, InputFeatures
from knowledgeextractor import KGEConfig
import tensorflow as tf
import numpy as np
from tensorflow.contrib import data as contrib_data
import collections

class TextProcessor(object):
    def processText(self, query_data):
        raise NotImplementedError

class MRCTextProcessor:
    def __init__(self, config_file):
        config=KGEConfig(config_file=config_file)
        self.tokenizer=tokenization.FullTokenizer( config.vocab )
        self.query_length=config.query_length
        self.max_sequence_length=config.max_sequence_length
        self.doc_stride=config.doc_stride

    def processText(self,query_data):
        example=self._build_example(query_data)
        feaure=self._convert_example_to_feature(example)
        return feaure

    def _build_example(self, entry):
        """Read a SQuAD style json data into a SquadExample."""
        '''
        {
            "id":...,
            "question":xxx,
            "context":xxx
        }
        '''

        paragraph_text = entry["context"]

        qas_id = entry["id"]
        question_text = entry["question"]
        start_position = None
        orig_answer_text = None
        is_impossible = False

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible)

        return example

    def _convert_example_to_feature(self, example):
        features=[]
        def call_backfunc(feature):
            features.append(feature)
        examples=[example]
        tokenizer=self.tokenizer
        max_seq_length=self.max_sequence_length
        doc_stride=self.doc_stride
        max_query_length=self.max_query_length
        is_training=False
        do_lower_case=False
        convert_examples_to_features(examples, tokenizer, max_seq_length,
                                doc_stride, max_query_length, is_training,
                                call_backfunc, do_lower_case)
        
        assert len(features)==1
        return features[0]

class SquenceLabelingTextProcessor(object):
    def __init__(self, config_file):
        self.config=KGEConfig(config_file=config_file)
        config=KGEConfig(config_file=config_file)
        self.tokenizer=tokenization.FullTokenizer( config.vocab )
        self.max_sequence_length=config.max_sequence_length
        self.doc_stride=config.doc_stride
        self.label_list=list(config.label_list)
        self.label_map={label:index for index, label in enumerate(self.label_list)}
        self.inv_label_map={v:k for k, v in self.label_map.items()}
    def convert_ids_labels(self, label_ids):
        labels=[]
        print("inv label map", self.inv_label_map)
        for label_id in label_ids:
            labels.append(self.inv_label_map[label_id])
        return labels

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