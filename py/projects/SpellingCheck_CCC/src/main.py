# -*- coding: utf-8 -*-
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from enchant import DictWithPWL
from nltk import tokenize
from enchant.checker import SpellChecker
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation

from UFUtils.CCC import Model
from UFUtils.spell_corrector import SpellCorrector
from UFUtils.utils import generate_ids, get_tokenizer, load_data

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BERT_VOCAB = '/home/giuseppe/PycharmProjects/SpellingCheck/model/vocab.txt'
BERT_INIT_CHKPNT = '../model/bert_model.ckpt'


def correct(possible_states, text_mask):
    tokenizer = get_tokenizer(BERT_VOCAB, BERT_INIT_CHKPNT)
    ops.reset_default_graph()
    sess = tf.InteractiveSession()
    model = Model()

    sess.run(tf.global_variables_initializer())
    var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bert')
    cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cls')
    saver = tf.train.Saver(var_list=var_lists + cls)
    saver.restore(sess, BERT_INIT_CHKPNT)

    replaced_masks = [text_mask.replace('**mask**', state) for state in possible_states]

    ids = [generate_ids(mask, tokenizer) for mask in replaced_masks]
    tokens, input_ids, tokens_ids = list(zip(*ids))

    indices, ids = [], []
    for i in range(len(input_ids)):
        indices.extend([i] * len(input_ids[i]))
        ids.extend(input_ids[i])
    masked_padded = tf.keras.preprocessing.sequence.pad_sequences(ids, padding='post')
    preds = sess.run(tf.nn.log_softmax(model.logits), feed_dict={model.X: masked_padded})

    indices = np.array(indices)
    scores = []

    for i in range(len(tokens)):
        filter_preds = preds[indices == i]
        total = []
        for k, x in enumerate(tokens_ids[i]):
            try:
                total.append(filter_preds[k, k + 1, x])
            except:
                total.append(filter_preds[k, k, x])
        scores.append(np.sum(total))

    prob_scores = np.array(scores) / np.sum(scores)
    probs = list(zip(possible_states, prob_scores))
    probs.sort(key=lambda x: x[1])

    return probs[0][0]


def replace_erros(dic, text):
    for k, v in dic.items():
        if has_lower(k):
            v = v.title()
        text = text.replace(k, v)
    return text


def has_lower(s):
    for c in s:
        if not c.islower():
            return True
    return False


def main(text):
    words = load_data('../data/counts_1grams.txt')
    corrector = SpellCorrector(words)
    # modificar para todas as palavras

    my_dict = DictWithPWL("en_US", "mywords.txt")
    my_checker = SpellChecker(my_dict)

    my_checker.set_text(text)
    text_tmp = deepcopy(text)
    corrected = {}
    for error in my_checker:
        err = error.word
        print(err)
        possible_states = corrector.edit_candidates(err.lower())
        mask = text_tmp.replace(err, '**mask**')
        correted_letter = correct(possible_states, mask)
        corrected.update({err: correted_letter})
    return replace_erros(corrected, text)


if __name__ == '__main__':
    text = "Smiarter Shopping. Bettar Livingt."
    sentences = tokenize.sent_tokenize(text)
    all_senteces = []
    for sentece in sentences:
        correct_sentece = main(sentece)
        all_senteces.append(correct_sentece)
    correct_sentece = ' '.join(all_senteces)
    print(">>>>>>>>>> BEFORE: ", sentece)
    print(">>>>>>>>>> AFTER: ", correct_sentece)
