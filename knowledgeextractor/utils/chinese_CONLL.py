"""Utilities for preprocessing and iterating over the CoNLL-2003-style Chinese words data.

"""

import re
from collections import defaultdict
import numpy as np
import tensorflow as tf


# pylint: disable=invalid-name, too-many-locals

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

UNK_WORD, UNK_CHAR, UNK_NER = 0, 0, 0
PAD_WORD, PAD_CHAR, PAD_NER = 1, 1, 1

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(r"\d")


def create_vocabs(train_path, dev_path, test_path, normalize_digits=True, min_occur=1, glove_dict=None):
    word_vocab = defaultdict(lambda: len(word_vocab))
    word_count = defaultdict(lambda: 0)
    ner_vocab = defaultdict(lambda: len(ner_vocab))

    UNK_WORD = word_vocab["<unk>"]
    PAD_WORD = word_vocab["<pad>"]
    UNK_NER = ner_vocab["<unk>"]
    PAD_NER = ner_vocab["<pad>"]

    print("Creating Vocabularies:")

    for file_path in [train_path, dev_path, test_path]:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(' ')

                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                ner = tokens[-1]

                if glove_dict is not None and (word in glove_dict or word.lower() in glove_dict):
                    word_count[word] += min_occur + 1
                elif file_path == train_path:
                    word_count[word] += 1

                nid = ner_vocab[ner]

    print("Total Vocabulary Size: %d" % len(word_count))
    for word in word_count:
        if word_count[word] > min_occur:
            wid = word_vocab[word]

    print("Word Vocabulary Size: %d" % len(word_vocab))
    print("NER Alphabet Size: %d" % len(ner_vocab))

    word_vocab = defaultdict(lambda: UNK_WORD, word_vocab)
    ner_vocab = defaultdict(lambda: UNK_NER, ner_vocab)

    i2w = {v: k for k, v in word_vocab.items()}
    i2n = {v: k for k, v in ner_vocab.items()}
    return (word_vocab, ner_vocab), (i2w, i2n)


def read_data(source_path, word_vocab, ner_vocab, normalize_digits=True, max_seq_length=256):
    data = []
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLReader(source_path, word_vocab, ner_vocab)
    
    #inst = reader.nextParagraphNerInst(normalize_digits, max_seq_length)
    inst=reader.nextSentenceNerInst(normalize_digits)
    
    while inst is not None:
        counter += 1
        sent = inst.sentence
        data.append([sent.word_ids,  inst.ner_ids])
        inst = reader.nextParagraphNerInst(normalize_digits, max_seq_length)

    reader.close()
    print("Total number of data: %d" % counter)
    return data


def iterate_batch(data, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(data)

    for start_idx in range(0, len(data), batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        batch = data[excerpt]

        batch_length = max([len(batch[i][0]) for i in range(len(batch))])

        wid_inputs = np.empty([len(batch), batch_length], dtype=np.int64)
        nid_inputs = np.empty([len(batch), batch_length], dtype=np.int64)
        masks = np.zeros([len(batch), batch_length], dtype=np.float32)
        lengths = np.empty(len(batch), dtype=np.int64)

        for i, inst in enumerate(batch):
            wids, nids = inst

            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_WORD
            
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_NER
            masks[i, :inst_size] = 1.0

        yield wid_inputs, nid_inputs, masks, lengths


def load_glove(filename, emb_dim, normalize_digits=True):
    """Loads embeddings in the glove text format in which each line is
    '<word-string> <embedding-vector>'. Dimensions of the embedding vector
    are separated with whitespace characters.

    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.

    Returns:
        The updated :attr:`word_vecs`.
    """
    glove_dict = dict()
    with tf.gfile.Open(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            word = tf.compat.as_text(word)
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            glove_dict[word] = np.array([float(v) for v in vec])
            if len(vec) != emb_dim:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (len(vec), emb_dim))
    return glove_dict


def construct_init_word_vecs(vocab, word_vecs, glove_dict):
    for word, index in vocab.items():
        if word in glove_dict:
            embedding = glove_dict[word]
        elif word.lower() in glove_dict:
            embedding = glove_dict[word.lower()]
        else:
            embedding = None

        if embedding is not None:
            word_vecs[index] = embedding
    return word_vecs


class CoNLLReader(object):
    def __init__(self, file_path, word_vocab, ner_vocab):
        self.__source_file = open(file_path, 'r', encoding='utf-8')
        self.__word_vocab = word_vocab
        self.__ner_vocab = ner_vocab

        self.previous_ner_inst=None

    def close(self):
        self.__source_file.close()

    def nextSentenceNerInst(self, normalize_digits=True):
        #each line is like : token tag, the blank line seprate the sentence
        # this method will fetch a senten ce each call
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            

            word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
            ner = tokens[-1]

            words.append(word)
            word_ids.append(self.__word_vocab[word])

            ner_tags.append(ner)
            ner_ids.append(self.__ner_vocab[ner])

        return NERInstance(Sentence(words, word_ids), ner_tags, ner_ids)

    def nextParagraphNerInst(self,normalize_digits=True, max_seq_length=256):
        nerInsts=[]
        curLen=0
        if self.previous_ner_inst is not None:
            nerInsts.append(self.previous_ner_inst)
            self.previous_ner_inst=None
        
            curLen=nerInsts[0].length()

        while curLen<max_seq_length:
            inst=self.nextSentenceNerInst()
            if inst is None:
                if curLen==0:
                    return None
                break
            nerInsts.append(inst)
            curLen+=inst.length()
        if curLen>max_seq_length and len(nerInsts)>1:
            self.previous_ner_inst=nerInsts[-1]
            nerInsts.pop()
            
        words=[]
        word_ids=[]
        ner_tags=[]
        ner_ids=[]
        for inst in nerInsts:
            words.extend(inst.sentence.words)
            word_ids.extend(inst.sentence.word_ids)
            ner_tags.extend(inst.ner_tags)
            ner_ids.extend(inst.ner_ids)
        

        return NERInstance(Sentence(words[:max_seq_length], word_ids[:max_seq_length]),
                    ner_tags[:max_seq_length], ner_ids[:max_seq_length])

class NERInstance(object):
    def __init__(self, sentence, ner_tags, ner_ids):
        self.sentence = sentence
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()


class Sentence(object):
    def __init__(self, words, word_ids):
        self.words = words
        self.word_ids = word_ids

    def length(self):
        return len(self.words)




#Writer Object
class CoNLLWriter(object):
    def __init__(self, i2w, i2n):
        self.__source_file = None
        self.__i2w = i2w
        self.__i2n = i2n

    def start(self, file_path):
        self.__source_file = open(file_path, 'w', encoding='utf-8')

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__i2w[word[i, j]]
                tgt = self.__i2n[targets[i, j]]
                pred = self.__i2n[predictions[i, j]]
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, "_", "_", tgt, pred))
            self.__source_file.write('\n')
