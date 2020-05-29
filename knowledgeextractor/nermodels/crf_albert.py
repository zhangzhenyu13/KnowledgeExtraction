from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import csv
import os
from albert import fine_tuning_utils
from albert import modeling
from albert import optimization
from albert import tokenization
from tensorflow import contrib as tf_contrib
from albert import classifier_utils
from knowledgeextractor import KGEConfig
from knowledgeextractor.utils import crf_utils, crf_processor

import tensorflow as tf
import logging

logging.getLogger(__name__).setLevel(logging.INFO)
    
def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     optimizer="adamw"):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn`."""

        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, probabilities, logits, predictions) = \
            create_model(albert_config, is_training, input_ids, input_mask,
                        segment_ids, label_ids, num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                False, optimizer)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op) 

        elif mode == tf.estimator.ModeKeys.EVAL:
            
            corrects = tf.reduce_sum(tf.cast(tf.equal(labels, predictions), tf.float32) * input_mask)
            accuracy=corrects/tf.reduce_sum(input_mask)
                
            eval_metrics= {
                    "eval_accuracy": accuracy,
                    "eval_loss": total_loss,
                }
            
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics) 
            
        else:
            
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "probabilities": probabilities,
                    "predictions": predictions
                }
            )
        return output_spec

    return model_fn


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """Creates a classification model."""
    (output_layer, enc_outputs) = fine_tuning_utils.create_albert(
        albert_config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=None,
        use_einsum=True,
        hub_module=None)


    (loss, probabilities, predictions)=[None]*3

    with tf.variable_scope("project"):
        """
        hidden layer between lstm layer and logits
        :param enc output: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        enc_output_shape=tf.shape(enc_outputs)
        hidden_size = output_layer.shape[-1].value

        # project to score of tags
        # B*T*D--> B*T*C
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[hidden_size, num_labels],
                                dtype=tf.float32, initializer=tf_contrib.layers.xavier_initializer())
            
            b = tf.get_variable("b", shape=[num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.nn.xw_plus_b(enc_outputs, W, b)
            logits= tf.reshape(pred, tf.concat([enc_output_shape[0:2], [num_labels]], axis=0) )
            probabilities = tf.nn.softmax(logits, axis=-1)
            predictions = tf.argmax(probabilities, axis=-1)
    
    if is_training:
        with tf.variable_scope("crf_loss"):
            """
            calculate crf loss
            :param project_logits: [1, num_steps, num_tags]
            :return: scalar loss
            """
            #'''
            trans = tf.get_variable(
                    "transitions",
                    shape=[num_labels, num_labels],
                    initializer=tf_contrib.layers.xavier_initializer())
            #'''
            lengths=tf.reduce_sum(input_mask, axis=-1)
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=lengths
                )

            loss = tf.reduce_mean(-log_likelihood)

    return (loss, probabilities, logits, predictions)


class NERModel(object):
        
    def __init__(self, config_file):
        self.config=KGEConfig(config_file)
        self.batch_size=self.config.batch_size
        self.processor=crf_processor.SquenceLabelingTextProcessor(self.config.processor)

        albert_config = modeling.AlbertConfig.from_json_file(
            self.config.albert_config_file)
        if self.config.max_seq_length > albert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the ALBERT model "
                "was only trained up to sequence length %d" %
                (self.config.max_seq_length, albert_config.max_position_embeddings))
        
        model_fn = model_fn_builder(
            albert_config=albert_config,
            num_labels=len(self.processor.label_list),
            init_checkpoint=None, learning_rate=1e-5, num_train_steps=1000,num_warmup_steps=100 # default hup
            )

        config = tf.ConfigProto(
            allow_soft_placement=True,#log_device_placement=True,
            gpu_options={"allow_growth":self.config.allow_growth}
            )
        
        run_config = tf.estimator.RunConfig(
            session_config=config)  
        
        self.estimator= tf.estimator.Estimator(
            model_fn=model_fn, #model_dir=self.config.saved_checkpoint,
            config=run_config, params={"batch_size":self.config.batch_size}
        )
    

    def predict(self, query_data_list):
        predict_features=[self.processor.processText(query) for query in query_data_list]

        logging.info("***** Running prediction*****")
        #predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn = crf_utils.data_based_input_fn_builder(
            features=predict_features,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = self.estimator.predict(
            input_fn=predict_input_fn
            ,checkpoint_path=self.config.saved_checkpoint 
            )

        
        return result

def test_init(*argv):
    global nm
    nm=NERModel("/home/zhangzy/KnowledgeExtraction/config/crf_albert_model.json")
    print("*"*100)
def test_predict(*args):
    
    query_data_list=args[0]
    res2=[]
    res=nm.predict(query_data_list)
    print(res)
    for re in res:
        print(re)
        res2.append(re)
    print("_"*100)

    return res2



if __name__ == "__main__":
    query_data={
        "guid":"test1",
        "text":"术中探查见盆腹腔未见腹水，子宫增大，约10*8*7CM，饱满，两侧附件未见异常，盆腔及腹主动脉旁淋巴结未及肿大。"
    }

    test_init()
    tokenizer=nm.processor.tokenizer
    toks=tokenizer.tokenize(query_data["text"])
    print(toks)
    print(query_data["text"])
    
    #exit(-1)

    res=test_predict([query_data])
    predictions=res[0]["predictions"]
    feature=nm.processor.processText(query_data)
    inputids=feature.input_ids
    words, labels= nm.processor.recover_words_tags(predictions,inputids)

    print(words)
    print(labels)