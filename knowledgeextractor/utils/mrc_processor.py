from albert.squad_utils import  SquadExample, InputFeatures
from knowledgeextractor.utils.mrc_utils import convert_single_example_to_features
from albert import tokenization
from knowledgeextractor import KGEConfig
import tensorflow as tf
import numpy as np
from tensorflow.contrib import data as contrib_data
import collections
from knowledgeextractor.utils.processor import TextProcessor
import six

class MRCTextProcessor(TextProcessor):
    def __init__(self, config_file):
        config=KGEConfig(config_file=config_file)
        self.tokenizer=tokenization.FullTokenizer( config.vocab )
        self.max_query_length=config.max_query_length
        self.max_sequence_length=config.max_sequence_length
        self.doc_stride=config.doc_stride

    def processText(self,query_data):
        example=self.build_example(query_data)
        feaure=self.convert_example_to_feature(example)
        return feaure

    def build_example(self, entry):
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

    def convert_example_to_features(self, example, ex_index):
        #features=[]
        #def call_backfunc(feature):
        #    features.append(feature)
        
        tokenizer=self.tokenizer
        max_seq_length=self.max_sequence_length
        doc_stride=self.doc_stride
        max_query_length=self.max_query_length
 
        features=convert_single_example_to_features(example, ex_index, tokenizer,
            max_seq_length, doc_stride,max_query_length)

        assert len(features)>0
        return features

def data_based_input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_p_masks = []
    all_start_positions = []
    all_end_positions = []
    all_is_impossibles=[]
    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_p_masks.append(feature.p_mask)
        all_start_positions.append(feature.start_position)
        all_end_positions.append(feature.end_position)
        all_is_impossibles.append(feature.is_impossible)


    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(
                    all_unique_ids, shape=[num_examples,],
                    dtype=tf.int32),
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
            "p_mask":
                tf.constant(all_p_masks, shape=[num_examples, seq_length], dtype=tf.int32),
            
            
            
        })

        '''
        "start_position":
                tf.constant(
                    all_start_positions,
                    shape=[num_examples, ],
                    dtype=tf.int32),
            "end_position":
                tf.constant(
                    all_end_positions,
                    shape=[num_examples, ],
                    dtype=tf.int32),
            "is_impossible":
                tf.constant(
                    all_is_impossibles,
                    shape=[num_examples,],
                    dtype=tf.int32),
        '''

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


