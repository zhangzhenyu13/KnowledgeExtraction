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
from knowledgeextractor.utils import crf_processor

import tensorflow as tf
    
def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw"):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, probabilities, logits, predictions) = \
            create_model(albert_config, is_training, input_ids, input_mask,
                        segment_ids, label_ids, num_labels, use_one_hot_embeddings,
                        task_name, hub_module)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                use_tpu, optimizer)
            '''
            output_spec = contrib_tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
            '''
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op) 

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                
                corrects = tf.reduce_sum(tf.cast(tf.equal(labels, predictions), tf.float32) * input_mask)
                accuracy=corrects/tf.reduce_sum(input_mask)
                loss = tf.metrics.mean(
                    values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }
            '''
            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = contrib_tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
            '''
            eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics) 
            
        else:
            '''
            output_spec = contrib_tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "probabilities": probabilities,
                    "predictions": predictions
                },
                scaffold_fn=scaffold_fn)
            '''
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
                 labels, num_labels, use_one_hot_embeddings, task_name,
                 hub_module):
    """Creates a classification model."""
    (output_layer, enc_outputs) = fine_tuning_utils.create_albert(
        albert_config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        use_einsum=True,
        hub_module=hub_module)


    (loss, per_example_loss, 
        probabilities, logits, predictions)=[None]*5

    with tf.variable_scope("crf_layer"):
        """
        hidden layer between lstm layer and logits
        :param enc output: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        enc_output_shape=tf.shape(enc_outputs)
        hidden_size = output_layer.shape[-1].value

        with tf.variable_scope("project"):
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

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    return (loss, per_example_loss, probabilities, logits, predictions)


class NERModel(object):
    def __init__(self, config_file):
        self.config=KGEConfig(config_file)
        self.batch_size=self.config.predict_batch_size
        self.processor=crf_processor.SquenceLabelingTextProcessor(self.config.processor)

        '''
        tpu_cluster_resolver = None
        if self.config.use_tpu and self.config.tpu_name:
            tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
                self.config.tpu_name, zone=self.config.tpu_zone, project=self.config.gcp_project)
        is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2

        run_config = contrib_tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.config.master,
            model_dir=self.config.output_dir,
            #save_checkpoints_steps=int(self.config.save_checkpoints_steps),
            #keep_checkpoint_max=0,
            #tpu_config=contrib_tpu.TPUConfig(
            #    iterations_per_loop=self.config.iterations_per_loop,
            #    num_shards=self.config.num_tpu_cores,
            #    per_host_input_for_training=is_per_host)
            )
        '''

        albert_config = modeling.AlbertConfig.from_json_file(
            self.config.albert_config_file)
        if self.config.max_seq_length > albert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the ALBERT model "
                "was only trained up to sequence length %d" %
                (self.config.max_seq_length, albert_config.max_position_embeddings))
        
        model_fn = model_fn_builder(
            albert_config=albert_config,
            num_labels=self.config.num_labels,
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=self.config.learning_rate,
            num_train_steps=self.config.train_step,
            num_warmup_steps=self.config.warmup_step,
            use_tpu=self.config.use_tpu,
            use_one_hot_embeddings=self.config.use_tpu,
            task_name="",
            hub_module=self.config.albert_hub_module_handle,
            #optimizer=self.config.optimizer
            )
        '''
        self.estimator = contrib_tpu.TPUEstimator(
            model_dir=self.config.output_dir,
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            #train_batch_size=self.config.train_batch_size,
            #eval_batch_size=self.config.eval_batch_size,
            predict_batch_size=self.config.predict_batch_size)
        '''
        config = tf.ConfigProto(
            allow_soft_placement=True,log_device_placement=True,
            gpu_options={"allow_growth":self.config.use_gpu})
        
        run_config = tf.estimator.RunConfig(
            session_config=config,
            model_dir=self.config.output_dir)        
        
        self.estimator= tf.estimator.Estimator(
            model_fn=model_fn, model_dir=self.config.output_dir,
            config=run_config, params={"batch_size":self.config.predict_batch_size}
        )
    

    def predict(self, query_data_list):
        num_actual_predict_examples = len(query_data_list)

        if self.config.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(query_data_list) % self.batch_size != 0:
                query_data_list.append({"guid":"Drop", "text": 5*"NaN "})

        predict_features=[self.processor.processText(query) for query in query_data_list]

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_features), num_actual_predict_examples,
                        len(predict_features) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.config.predict_batch_size)

        predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn = crf_processor.data_based_input_fn_builder(
            features=predict_features,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        #checkpoint_path = os.path.join(self.output_dir, "model.ckpt-best")
        result = self.estimator.predict(
            input_fn=predict_input_fn)#,checkpoint_path=checkpoint_path)

        #result=self.sess.run(result)
        
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
    
    exit(-1)

    res=test_predict([query_data])
    predictions=res[0]["predictions"]
    feature=nm.processor.processText(query_data)
    inputids=feature.input_ids
    words, labels= nm.processor.recover_token_tags(predictions,inputids)

    print(words)
    print(labels)