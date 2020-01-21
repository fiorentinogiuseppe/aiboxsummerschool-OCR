# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from enchant import DictWithPWL
from enchant.checker import SpellChecker
from .UFUtils.spell_corrector import SpellCorrector
from .UFUtils.model import Model
from .UFUtils.utils import generate_ids, tokens_to_masked_ids, load_data
from bert import tokenization

import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
BERT_INIT_CHKPNT = '../model/uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_VOCAB = '../model/uncased_L-12_H-768_A-12/vocab.txt'

tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(
    vocab_file=BERT_VOCAB, do_lower_case=True)

corrector = SpellCorrector()

# TODO modificar para todas as palavras

my_dict = DictWithPWL("en_US", "mywords.txt")
my_checker = SpellChecker(my_dict)
text = "This is sme sample txt with erors."
my_checker.set_text(text)
text_mask = deepcopy(text)

for error in my_checker:
    err = error.word
    possible_states = corrector.edit_candidates(err)
    print(possible_states)
    text_mask = text_mask.replace(err, '**mask**')



tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model()

sess.run(tf.global_variables_initializer())
var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bert')

cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cls')
cls

saver = tf.train.Saver(var_list=var_lists + cls)
saver.restore(sess, BERT_INIT_CHKPNT)

replaced_masks = [text_mask.replace('**mask**', state) for state in possible_states]
replaced_masks

tokens = tokenizer.tokenize(replaced_masks[0])
input_ids = [tokens_to_masked_ids(tokens, i) for i in range(len(tokens))]
input_ids

tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
tokens_ids

ids = [generate_ids(mask) for mask in replaced_masks]
tokens, input_ids, tokens_ids = list(zip(*ids))

indices, ids = [], []
for i in range(len(input_ids)):
    indices.extend([i] * len(input_ids[i]))
    ids.extend(input_ids[i])

ids[0]

masked_padded = tf.keras.preprocessing.sequence.pad_sequences(ids, padding='post')
masked_padded.shape

preds = sess.run(tf.nn.log_softmax(model.logits), feed_dict={model.X: masked_padded})
preds.shape

indices = np.array(indices)
scores = []

for i in range(len(tokens)):
    filter_preds = preds[indices == i]
    total = np.sum([filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])])
    scores.append(total)

scores

prob_scores = np.array(scores) / np.sum(scores)
prob_scores

probs = list(zip(possible_states, prob_scores))
probs.sort(key=lambda x: x[1])

corrected = text_mask.replace('**mask**', probs[0][0])

corrected
