"""Sequence tagging.
"""

import os
import time
import importlib
import numpy as np
import tensorflow as tf
import texar.tf as tx
from albert.lamb_optimizer import LAMBOptimizer
from knowledgeextractor.utils.scores import scores
from knowledgeextractor.utils.chinese_CONLL import (create_vocabs, create_vocabs_from_vocab_file, \
    read_data, iterate_batch, CoNLLWriter)
from knowledgeextractor.utils import train_utils
from albert.fine_tuning_utils import create_albert
from albert.tokenization import FullTokenizer
from albert.modeling import AlbertModel, AlbertConfig
from albert import modeling
import logging
logging.getLogger().setLevel(logging.INFO)
flags = tf.flags

flags.DEFINE_string("data_path", "./data",
                    "Directory containing NER data (e.g., eng.train.bio.conll).")
flags.DEFINE_string("train", "eng.train.bio.conll",
                    "the file name of the training data.")
flags.DEFINE_string("dev", "eng.dev.bio.conll",
                    "the file name of the dev data.")
flags.DEFINE_string("test", "eng.test.bio.conll",
                    "the file name of the test data.")
flags.DEFINE_string("checkpoint", "path_to_checkpoint",
                    "the file name of the albert checkpoint.")

flags.DEFINE_string("albert_config", "",
                    "the file name of the albert config file.")

flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

train_path = os.path.join(FLAGS.data_path, FLAGS.train)
dev_path = os.path.join(FLAGS.data_path, FLAGS.dev)
test_path = os.path.join(FLAGS.data_path, FLAGS.test)
checkpoint_path =  FLAGS.checkpoint
albert_config_path= FLAGS.albert_config
# Prepares/loads data
vocab_file=config.vocab
label_file=config.label
albert_config=AlbertConfig.from_json_file(albert_config_path)

(word_vocab, ner_vocab), (i2w, i2n) = create_vocabs_from_vocab_file(vocab_file, label_file)
#(word_vocab, ner_vocab), (i2w, i2n) = create_vocabs(train_path, dev_path, test_path)

data_train = read_data(train_path, word_vocab, ner_vocab)
data_dev = read_data(dev_path, word_vocab, ner_vocab)
data_test = read_data(test_path, word_vocab, ner_vocab)




# Builds TF graph
inputs = tf.placeholder(tf.int64, [None, None])
targets = tf.placeholder(tf.int64, [None, None])
masks = tf.placeholder(tf.float32, [None, None])
seq_lengths = tf.placeholder(tf.int64, [None])


vocab_size = len(word_vocab)


# Source word embedding# def get
encoder=AlbertModel(albert_config,is_training=True,input_ids=inputs, input_mask=masks)

outputs=encoder.get_sequence_output()

def project_enc_layer(enc_outputs, name=None):
    """
    hidden layer between lstm layer and logits
    :param enc output: [batch_size, num_steps, emb_size]
    :return: [batch_size, num_steps, num_tags]
    """
    enc_output_shape=tf.shape(enc_outputs)
    with tf.variable_scope("project" if not name else name):
        # project to score of tags
        # B*T*D--> B*T*C
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[config.hidden_dim, len(ner_vocab)],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            
            b = tf.get_variable("b", shape=[len(ner_vocab)], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.nn.xw_plus_b(enc_outputs, W, b)
        return tf.reshape(pred, tf.concat([enc_output_shape[0:2], [len(ner_vocab)]], axis=0) )

logits=project_enc_layer(enc_outputs=outputs)


def crf_layer_loss(logits, labels=None):
    """
    calculate crf loss
    :param project_logits: [1, num_steps, num_tags]
    :return: scalar loss
    """

    with tf.variable_scope("crf_loss"):
        
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=labels,
            #transition_params=trans,
            sequence_lengths=seq_lengths)
        return tf.reduce_mean(-log_likelihood), trans

crf_loss, _=crf_layer_loss(logits, targets)

mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    labels=targets,
    logits=logits,
    sequence_length=seq_lengths,
    average_across_batch=True,
    average_across_timesteps=True,
    sum_over_timesteps=False)

predicts = tf.argmax(logits, axis=2)
corrects = tf.reduce_sum(tf.cast(tf.equal(targets, predicts), tf.float32) * masks)



global_step = tf.placeholder(tf.int32)

init_lr=1e-3
poly_power=1.0
learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
num_train_steps=config.num_epochs*len(data_train)//config.batch_size
# Implements linear decay of the learning rate.
learning_rate = tf.train.polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate=0.0,
    power=poly_power,
    cycle=False)
optimizer=LAMBOptimizer(
    learning_rate=learning_rate,
    weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )

train_op = tx.core.get_train_op(
    crf_loss, global_step=global_step, 
    increment_global_step=False,
    optimizer=optimizer,
    #learning_rate=learning_rate,
    #hparams=config.opt
    )

# Training/eval processes

def _train_epoch(sess, epoch, step):
    start_time = time.time()
    loss = 0.
    corr = 0.
    num_tokens = 0.

    fetches = {
        "mle_loss": mle_loss,
        "correct": corrects,
    }
    fetches["train_op"] = train_op

    mode = tf.estimator.ModeKeys.TRAIN
    num_inst = 0
    for batch in iterate_batch(data_train, config.batch_size, shuffle=True):
        word, ner, mask, length = batch
        feed_dict = {
            inputs: word, targets: ner, masks: mask, seq_lengths: length,
            global_step: epoch, tx.global_mode(): mode,
            learning_rate: train_utils.get_lr(step, config.lr)
        }

        rets = sess.run(fetches, feed_dict)
        nums = np.sum(length)
        num_inst += len(word)
        loss += rets["mle_loss"] * nums
        corr += rets["correct"]
        num_tokens += nums

        step+=1

        print("train: %d (%d/%d) loss: %.4f, acc: %.2f%%" % (epoch, num_inst, len(data_train), loss / num_tokens, corr / num_tokens * 100))
    print("train: %d loss: %.4f, acc: %.2f%%, time: %.2fs" % (epoch, loss / num_tokens, corr / num_tokens * 100, time.time() - start_time))

    return step

def _eval(sess, epoch, data_tag):
    fetches = {
        "predicts": predicts,
    }
    mode = tf.estimator.ModeKeys.EVAL
    file_name = 'tmp/%s%d' % (data_tag, epoch)
    
    writer = CoNLLWriter(i2w, i2n)
    writer.start(file_name)
    data = data_dev if data_tag == 'dev' else data_test
    for batch in iterate_batch(data, config.batch_size, shuffle=False):
        word, ner, mask, length = batch
        feed_dict = {
            inputs: word, targets: ner, masks: mask, seq_lengths: length,
            global_step: epoch, tx.global_mode(): mode,
        }
        rets = sess.run(fetches, feed_dict)
        predictions = rets['predicts']
        writer.write(word, predictions, ner, length)
    writer.close()
    acc, precision, recall, f1 = scores(file_name)
    print('%s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (data_tag, acc, precision, recall, f1))
    return acc, precision, recall, f1


#load checkpoint
tvars = tf.trainable_variables()
initialized_variable_names = {}

(assignment_map, initialized_variable_names
) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_path)

tf.train.init_from_checkpoint(checkpoint_path, assignment_map)

logging.info("**** Trainable Variables ****")
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    print( "  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

print("checkpoint loaded")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    

    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    best_epoch = 0

    test_f1 = 0.0
    test_acc = 0.0
    test_prec = 0.0
    test_recall = 0.0

    tx.utils.maybe_create_dir('./tmp')
    
    step=0

    for epoch in range(config.num_epochs):
        step=_train_epoch(sess, epoch, step)
        acc, precision, recall, f1 = _eval(sess, epoch, 'dev')
        if dev_f1 < f1:
            dev_f1 = f1
            dev_acc = acc
            dev_precision = precision
            dev_recall = recall
            best_epoch = epoch
            test_acc, test_prec, test_recall, test_f1 = _eval(sess, epoch, 'test')
        print('best acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%, epoch: %d' % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
        print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%, epoch: %d' % (test_acc, test_prec, test_recall, test_f1, best_epoch))
        print('---------------------------------------------------')
        
        saver=tf.train.Saver()
        saver.save(sess, "saved_ner_models/albert")
        print("saved model")

