from albert.squad_utils import SquadExample, InputFeatures
from albert import tokenization
from knowledgeextractor import KGEConfig
import tensorflow as tf
import numpy as np
import collections
import six
import math 

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
     "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs

SPIECE_UNDERLINE=u"##".encode("utf-8")
#----------convert example to feature(s)---------------
def _convert_index(index, pos, m=None, is_start=True):
  """Converts index."""
  if index[pos] is not None:
    return index[pos]
  n = len(index)
  rear = pos
  while rear < n - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if m is not None and index[front] < m - 1:
      if is_start:
        return index[front] + 1
      else:
        return m - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def convert_single_example_to_features(example:SquadExample, ex_index, 
     tokenizer:tokenization.FullTokenizer,
     max_seq_length, doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features=[]
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)

    query_tokens = tokenizer.tokenize(tokenization.preprocess_text( example.question_text) )

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    paragraph_text = example.paragraph_text
    para_tokens = tokenizer.tokenize( 
        tokenization.preprocess_text(example.paragraph_text,remove_space=True, lower=False) 
        )

    # I am a good man!
    # char tok 2 tok index 0 1 1 3 4 4 4 4 8 8 8 11
    # tok start to chartok index 0 1 3 4 8 11
    # tok end to chartok index 0 2 3 7 10 11
    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
    for i, token in enumerate(para_tokens):
        new_token = six.ensure_text(token).replace(
            SPIECE_UNDERLINE.decode("utf-8"), " ")
        chartok_to_tok_index.extend([i] * len(new_token))
        tok_start_to_chartok_index.append(char_cnt)
        char_cnt += len(new_token)
        tok_end_to_chartok_index.append(char_cnt - 1)
    
    # compare tok seq and the original text seq with LCS
    tok_cat_text = "".join(para_tokens).replace(
        SPIECE_UNDERLINE.decode("utf-8"), " ")
    n, m = len(paragraph_text), len(tok_cat_text)

    if n > max_n or m > max_m:
        max_n = max(n, max_n)
        max_m = max(m, max_m)
        f = np.zeros((max_n, max_m), dtype=np.float32)

    g = {}

    def _lcs_match(max_dist, n=n, m=m):
        """Longest-common-substring algorithm."""
        f.fill(0)
        g.clear()

        ### longest common sub sequence
        # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
        for i in range(n):

            # note(zhiliny):
            # unlike standard LCS, this is specifically optimized for the setting
            # because the mismatch between sentence pieces and original text will
            # be small
            for j in range(i - max_dist, i + max_dist):
                if j >= m or j < 0: continue

                if i > 0:
                    g[(i, j)] = 0
                    f[i, j] = f[i - 1, j]

                if j > 0 and f[i, j - 1] > f[i, j]:
                    g[(i, j)] = 1
                    f[i, j] = f[i, j - 1]

                f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                if (tokenization.preprocess_text(
                    paragraph_text[i], remove_space=False, lower=False) == tok_cat_text[j]
                    and f_prev + 1 > f[i, j]):
                    g[(i, j)] = 2
                    f[i, j] = f_prev + 1

    max_dist = abs(n - m) + 5
    for _ in range(2):
        _lcs_match(max_dist)
        if f[n - 1, m - 1] > 0.8 * n: break
        max_dist *= 2

    orig_to_chartok_index = [None] * n
    chartok_to_orig_index = [None] * m
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        if (i, j) not in g: break
        if g[(i, j)] == 2:
            orig_to_chartok_index[i] = j
            chartok_to_orig_index[j] = i
            i, j = i - 1, j - 1
        elif g[(i, j)] == 1:
            j = j - 1
        else:
            i = i - 1

    if (all(v is None for v in orig_to_chartok_index) or
        f[n - 1, m - 1] < 0.8 * n):
        # the common seq chars are below the threshold (firstly,
        # check whether null com seq chars for efficiency)
        # those that differs too much are ignored
    #if all(v is None for v in orig_to_chartok_index):
        tf.logging.info("MISMATCH DETECTED!")
        '''
        print("--------------MISMATCH DETECTED!-----------------")
        print(f[n-1, m-1], n, 0.8*n)
        print([v for v in orig_to_chartok_index])
        print("p text:",paragraph_text)
        print("p tokens:",para_tokens)
        print("p processed-text:",tokenization.preprocess_text(paragraph_text, remove_space=False))
        print("tok cat text:",tok_cat_text)
        '''
        return  features

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
        start_chartok_pos = tok_start_to_chartok_index[i]
        end_chartok_pos = tok_end_to_chartok_index[i]
        start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                        n, is_start=True)
        end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                    n, is_start=False)

        tok_start_to_orig_index.append(start_orig_pos)
        tok_end_to_orig_index.append(end_orig_pos)

    #all_doc_tokens = tokenizer.convert_tokens_to_ids(para_tokens)
    all_doc_tokens=para_tokens
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        # construct the [CLS] query_doc [SEP] doc_span [SEP] input
        # p_mask for query and special tokens are 1 without any attending to it
        tokens = []
        token_is_max_context = {}
        segment_ids = []
        p_mask = []

        cur_tok_start_to_orig_index = []
        cur_tok_end_to_orig_index = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        p_mask.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            p_mask.append(1)
        tokens.append("[SEP]")
        segment_ids.append(0)
        p_mask.append(1)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i

            cur_tok_start_to_orig_index.append(
                tok_start_to_orig_index[split_token_index])
            cur_tok_end_to_orig_index.append(
                tok_end_to_orig_index[split_token_index])

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                    split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
            p_mask.append(0)

        tokens.append("[SEP]")
        segment_ids.append(1)
        p_mask.append(1)

        paragraph_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #print("---------tokens----------")
        #print(tokens)

        feature = InputFeatures(
            unique_id=example.qas_id,
            example_index=ex_index,
            doc_span_index=doc_span_index,
            tok_start_to_orig_index=cur_tok_start_to_orig_index,
            tok_end_to_orig_index=cur_tok_end_to_orig_index,
            token_is_max_context=token_is_max_context,
            tokens=tokens,#[tokenizer.sp_model.IdToPiece(x) for x in tokens],
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            paragraph_len=paragraph_len,
            p_mask=p_mask
            #start_position=start_position,
            #end_position=end_position,
            #is_impossible=span_is_impossible,
            )

        features.append(feature)


    return features

def get_predictions_v2(result_dict, cls_dict, all_examples, all_features,
                         all_results, n_best_size, max_answer_length,
                         null_score_diff_threshold):
  """ final predictions and log-odds of null if needed."""


  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    # score_null = 1000000  # large and positive

    for (feature_index, feature) in enumerate(features):
      for ((start_idx, end_idx), logprobs) in \
        result_dict[example_index][feature.unique_id].items():
        #each example can have several features due to doc stride
        start_log_prob = 0
        end_log_prob = 0
        for logprob in logprobs:
          start_log_prob += logprob[0]
          end_log_prob += logprob[1]
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index=start_idx,
                end_index=end_idx,
                start_log_prob=start_log_prob / len(logprobs),
                end_log_prob=end_log_prob / len(logprobs)))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      tok_start_to_orig_index = feature.tok_start_to_orig_index
      tok_end_to_orig_index = feature.tok_end_to_orig_index
      start_orig_pos = tok_start_to_orig_index[pred.start_index]
      end_orig_pos = tok_end_to_orig_index[pred.end_index]

      paragraph_text = example.paragraph_text
      final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(
              text="",
              start_log_prob=-1e6,
              end_log_prob=-1e6))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_log_prob"] = entry.start_log_prob
      output["end_log_prob"] = entry.end_log_prob
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None

    score_diff = sum(cls_dict[example_index]) / len(cls_dict[example_index])
    scores_diff_json[example.qas_id] = score_diff
    # predict null answers when null threshold is provided
    if null_score_diff_threshold is None or score_diff < null_score_diff_threshold:
      all_predictions[example.qas_id] = best_non_null_entry.text
    else:
      all_predictions[example.qas_id] = ""

    all_nbest_json[example.qas_id] = nbest_json
    assert len(nbest_json) >= 1

  return all_predictions, scores_diff_json, all_nbest_json