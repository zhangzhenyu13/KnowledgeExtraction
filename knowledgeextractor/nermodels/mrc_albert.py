from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from knowledgeextractor import KGEConfig
from knowledgeextractor.utils.mrc_processor import MRCTextProcessor, data_based_input_fn_builder
from knowledgeextractor.utils.mrc_utils import get_predictions_v2
import json
import os
import random
import time

from albert import fine_tuning_utils
from albert import modeling
from albert import squad_utils
import six
import tensorflow.compat.v1 as tf

from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu


class NERModel(object):
    def __init__(self, config_file):
        self.config=KGEConfig(config_file)
        self.processor=MRCTextProcessor(self.config.processor)
        self.debug=self.config.debug
        FLAGS=self.config

        albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
        
        iterations_per_loop = 1000

        run_config = contrib_tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.model_dir,
           # keep_checkpoint_max=0,
           # save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=contrib_tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host)
        )



        model_fn = squad_utils.v2_model_fn_builder(
            albert_config=albert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=1e-5,
            num_train_steps=1,
            num_warmup_steps=1,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu,
            max_seq_length=FLAGS.max_seq_length,
            start_n_top=FLAGS.start_n_top,
            end_n_top=FLAGS.end_n_top,
            dropout_prob=FLAGS.dropout_prob,
            hub_module=FLAGS.albert_hub_module_handle)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = contrib_tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            model_dir=FLAGS.model_dir
        )
        
        
    
    def predict(self, query_data_list):
        features=[]
        examples=[]
        for (ex_index, query_data) in enumerate(query_data_list):
            example=self.processor.build_example(query_data)
            feature_bs=self.processor.convert_example_to_features(example, ex_index)
            examples.append(example)
            
            #features.append(feature_bs[0])
            features.extend(feature_bs)
        
        predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn=data_based_input_fn_builder(
            features, self.config.max_seq_length, False, predict_drop_remainder
        )

        def get_result(checkpoint=None):
            """Evaluate the checkpoint on SQuAD v2.0."""
            # If running eval on the TPU, you will need to specify the number of
            # steps.
            all_results = []
            predictions=self.estimator.predict(
                predict_input_fn, yield_single_examples=True
                #checkpoint_path=checkpoint
            )

            for result in predictions:
                if len(all_results) % 1000 == 0:
                    tf.logging.info("Processing example: %d" % (len(all_results)))

                unique_id = int(result["unique_ids"])
                start_top_log_probs = (
                    [float(x) for x in result["start_top_log_probs"].flat])
                start_top_index = [int(x) for x in result["start_top_index"].flat]
                end_top_log_probs = (
                    [float(x) for x in result["end_top_log_probs"].flat])
                end_top_index = [int(x) for x in result["end_top_index"].flat]

                cls_logits = float(result["cls_logits"].flat[0])
                all_results.append(
                    squad_utils.RawResultV2(
                        unique_id=unique_id,
                        start_top_log_probs=start_top_log_probs,
                        start_top_index=start_top_index,
                        end_top_log_probs=end_top_log_probs,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits))

            return all_results

        all_results=get_result()
        result_dict = {}
        cls_dict = {}
        squad_utils.accumulate_predictions_v2(
            result_dict, cls_dict, examples, features,
            all_results, self.config.n_best_size, self.config.max_answer_length,
            self.config.start_n_top, self.config.end_n_top)


        predictions, na_probs, all_nbest_json = get_predictions_v2(
            result_dict, cls_dict, examples, features,
            all_results, self.config.n_best_size, self.config.max_answer_length,
            null_score_diff_threshold=None
        )
        
        if self.debug:    
            return result_dict, cls_dict, predictions, na_probs, all_nbest_json
        else:
            return all_nbest_json
'''-------------test---------------'''
def test():
    nm=NERModel("/home/zhangzy/KnowledgeExtraction/config/mrc_albert_model.json")
    result_dict, result_cls, predictions, na_probs, all_nbest_json =nm.predict([query_data])
    print(len(result_dict[0][9987]),result_dict)
    print("*"*100)
    print(result_cls)
    print("*"*100)
    print(predictions)
    print("*"*100)
    print(na_probs)
    print("*"*100)
    print(all_nbest_json)
    print("**100")
    print()

def run():
    nm=NERModel("/home/zhangzy/KnowledgeExtraction/config/mrc_albert_model.json")
    all_nbest_json =nm.predict([query_data])
    print("*"*100)
    print(all_nbest_json)
    print("*"*100)
    print()

if __name__ == "__main__":
    query_data={
        "question":"列出地名，例如河流、城市、山川等等",
        "context":"北京，是中华人民共和国的首都，几千年历史，有好几条非常有名的河流绕流而过。",
        "id": 9987
    }

    #test()
    
    run()


