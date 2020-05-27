'''the snippet is adpated from the google AI's ALBERT source code for training MRC on SQUAD v2.0'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import random
import time

from albert import fine_tuning_utils
from albert import modeling
from albert import squad_utils
import six
import tensorflow.compat.v1 as tf

from knowledgeextractor.utils import mrc_utils

# pylint: disable=g-import-not-at-top
if six.PY2:
  import six.moves.cPickle as pickle
else:
  import pickle
# pylint: enable=g-import-not-at-top

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string("train_feature_file", None,
                    "training feature file.")

flags.DEFINE_string(
    "predict_feature_file", None,
    "Location of predict features. If it doesn't exist, it will be written. "
    "If it does exist, it will be read.")

flags.DEFINE_string(
    "predict_feature_left_file", None,
    "Location of predict features not passed to TPU. If it doesn't exist, it "
    "will be written. If it does exist, it will be read.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")



flags.DEFINE_integer("start_n_top", 5, "beam size for the start positions.")

flags.DEFINE_integer("end_n_top", 5, "beam size for the end positions.")

flags.DEFINE_float("dropout_prob", 0.1, "dropout probability.")


def validate_flags_or_throw(albert_config):
  """Validate the input FLAGS or throw an exception."""

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")
    if not FLAGS.predict_feature_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_feature_file` must be "
          "specified.")
    if not FLAGS.predict_feature_left_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_feature_left_file` must be "
          "specified.")

  if FLAGS.max_seq_length > albert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the ALBERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, albert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

  validate_flags_or_throw(albert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, "cached_dir"))

  tokenizer = fine_tuning_utils.create_vocab(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      spm_model_file=None,
      hub_module=None)


  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  #train_examples = squad_utils.read_squad_examples(
  #    input_file=FLAGS.train_file, is_training=True)
  train_examples= mrc_utils.read_squad_examples(
      input_file=FLAGS.train_file, is_training=True
  )    
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  if FLAGS.do_train:
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

# construct estimator
  tf.logging.info("building the estimator ...")
  session_config = tf.ConfigProto(
    allow_soft_placement=True,log_device_placement=True,
    device_count={'GPU': 0} 
    )
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.9  
  session_config.gpu_options.allow_growth = True  
  session_config.gpu_options.allocator_type = 'BFC' 

  run_config = tf.estimator.RunConfig(
      session_config=session_config,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      log_step_count_steps=FLAGS.log_step_count_steps) 

  model_fn = squad_utils.v2_model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      max_seq_length=FLAGS.max_seq_length,
      start_n_top=FLAGS.start_n_top,
      end_n_top=FLAGS.end_n_top,
      dropout_prob=FLAGS.dropout_prob,
      hub_module=FLAGS.albert_hub_module_handle)

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,config=run_config,params={"batch_size":FLAGS.train_batch_size}
    )

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.

    if not tf.gfile.Exists(FLAGS.train_feature_file):
      train_writer = squad_utils.FeatureWriter(
          filename=os.path.join(FLAGS.train_feature_file), is_training=True)
      squad_utils.convert_examples_to_features(
          examples=train_examples,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=True,
          output_fn=train_writer.process_feature,
          do_lower_case=FLAGS.do_lower_case)
      train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    # tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = squad_utils.input_fn_builder(
        input_file=FLAGS.train_feature_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size,
        is_v2=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    with tf.gfile.Open(FLAGS.predict_file) as predict_file:
      prediction_json = json.load(predict_file)["data"]
    #eval_examples = squad_utils.read_squad_examples(
    #    input_file=FLAGS.predict_file, is_training=False)
    eval_examples = mrc_utils.read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    if (tf.gfile.Exists(FLAGS.predict_feature_file) and tf.gfile.Exists(
        FLAGS.predict_feature_left_file)):
      tf.logging.info("Loading eval features from {}".format(
          FLAGS.predict_feature_left_file))
      with tf.gfile.Open(FLAGS.predict_feature_left_file, "rb") as fin:
        eval_features = pickle.load(fin)
    else:
      eval_writer = squad_utils.FeatureWriter(
          filename=FLAGS.predict_feature_file, is_training=False)
      eval_features = []

      def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

      squad_utils.convert_examples_to_features(
          examples=eval_examples,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=False,
          output_fn=append_feature,
          do_lower_case=FLAGS.do_lower_case)
      eval_writer.close()

      with tf.gfile.Open(FLAGS.predict_feature_left_file, "wb") as fout:
        pickle.dump(eval_features, fout)

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = squad_utils.input_fn_builder(
        input_file=FLAGS.predict_feature_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.predict_batch_size,
        is_v2=True)

    def get_result(checkpoint):
      """Evaluate the checkpoint on SQuAD v2.0."""
      # If running eval on the TPU, you will need to specify the number of
      # steps.
      reader = tf.train.NewCheckpointReader(checkpoint)
      global_step = reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
      all_results = []
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True,
          checkpoint_path=checkpoint):
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

      output_prediction_file = os.path.join(
          FLAGS.output_dir, "predictions.json")
      output_nbest_file = os.path.join(
          FLAGS.output_dir, "nbest_predictions.json")
      output_null_log_odds_file = os.path.join(
          FLAGS.output_dir, "null_odds.json")

      result_dict = {}
      cls_dict = {}
      squad_utils.accumulate_predictions_v2(
          result_dict, cls_dict, eval_examples, eval_features,
          all_results, FLAGS.n_best_size, FLAGS.max_answer_length,
          FLAGS.start_n_top, FLAGS.end_n_top)

      return squad_utils.evaluate_v2(
          result_dict, cls_dict, prediction_json, eval_examples,
          eval_features, all_results, FLAGS.n_best_size,
          FLAGS.max_answer_length, output_prediction_file, output_nbest_file,
          output_null_log_odds_file), int(global_step)

    def _find_valid_cands(curr_step):
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      candidates = []
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          idx = ckpt_name.split("-")[-1]
          if idx != "best" and int(idx) > curr_step:
            candidates.append(filename)
      return candidates

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
    key_name = "f1"
    writer = tf.gfile.GFile(output_eval_file, "w")
    if tf.gfile.Exists(checkpoint_path + ".index"):
      result = get_result(checkpoint_path)
      best_perf = result[0][key_name]
      global_step = result[1]
    else:
      global_step = -1
      best_perf = -1
      checkpoint_path = None
    while global_step < num_train_steps:
      steps_and_files = {}
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
          if cur_filename.split("-")[-1] == "best":
            continue
          gstep = int(cur_filename.split("-")[-1])
          if gstep not in steps_and_files:
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files[gstep] = cur_filename
      tf.logging.info("found {} files.".format(len(steps_and_files)))
      if not steps_and_files:
        tf.logging.info("found 0 file, global step: {}. Sleeping."
                        .format(global_step))
        time.sleep(60)
      else:
        for ele in sorted(steps_and_files.items()):
          step, checkpoint_path = ele
          if global_step >= step:
            if len(_find_valid_cands(step)) > 1:
              for ext in ["meta", "data-00000-of-00001", "index"]:
                src_ckpt = checkpoint_path + ".{}".format(ext)
                tf.logging.info("removing {}".format(src_ckpt))
                tf.gfile.Remove(src_ckpt)
            continue
          result, global_step = get_result(checkpoint_path)
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tgt_ckpt = checkpoint_path.rsplit(
                  "-", 1)[0] + "-best.{}".format(ext)
              tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
              tf.gfile.Copy(src_ckpt, tgt_ckpt, overwrite=True)
              writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))
          writer.write("best {} = {}\n".format(key_name, best_perf))
          tf.logging.info("  best {} = {}\n".format(key_name, best_perf))

          if len(_find_valid_cands(global_step)) > 2:
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tf.logging.info("removing {}".format(src_ckpt))
              tf.gfile.Remove(src_ckpt)
          writer.write("=" * 50 + "\n")

    checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
    result, global_step = get_result(checkpoint_path)
    tf.logging.info("***** Final Eval results *****")
    for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))
    writer.write("best perf happened at step: {}".format(global_step))


if __name__ == "__main__":
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
