from albert.squad_utils import convert_examples_to_features, SquadExample
from albert import tokenization
from knowledgeextractor.utils.crf_utils import convert_single_example, InputExample, InputFeatures
from knowledgeextractor import KGEConfig
import tensorflow as tf
import numpy as np
from tensorflow.contrib import data as contrib_data
import collections
from knowledgeextractor.utils.processor import TextProcessor

class MRCTextProcessor(TextProcessor):
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