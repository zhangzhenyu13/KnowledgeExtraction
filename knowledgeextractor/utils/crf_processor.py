from albert import tokenization
from knowledgeextractor.utils.crf_utils import (convert_single_example, InputExample, 
      InputFeatures, load_labels)
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
        if config.label_list and type(config.label_list)==list:
          self.label_list=list(config.label_list)
        else:
          self.label_list=load_labels(config.label_list)
          
        self.label_map={label:index for index, label in enumerate(self.label_list)}
        self.inv_label_map={v:k for k, v in self.label_map.items()}
        
        self.piece_prefix="##"
        #print(self.label_list)
        #print(self.label_map)
        #print(self.inv_label_map)

    def recover_words_tags(self, label_ids, input_ids):
        labels=[]
        #print("inv label map", self.inv_label_map)
        for label_id in label_ids:
            labels.append(self.inv_label_map[label_id])
        
        words=self.tokenizer.convert_ids_to_tokens(input_ids)
        assert len(labels)==len(words)

        index=0 
        while index< len(labels):
          if words[index]=="[SEP]":
            break
          index+=1
        words=words[1:index]
        labels=labels[1:index]
        
        return words, labels

    def processText(self, query_data):
        '''
        {
            "guid":id,
            "text":....,

        }
        '''
        guid=query_data["guid"]
        text=query_data["text"]
        #text="".join( self.tokenizer.basic_tokenizer.tokenize(text) )
        example=InputExample(guid=guid, text=text)
        ex_index=0
        label_list=self.label_list
        max_seq_length=self.max_sequence_length
        tokenizer=self.tokenizer
        feature=convert_single_example(ex_index, example, max_seq_length,
                           self)
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