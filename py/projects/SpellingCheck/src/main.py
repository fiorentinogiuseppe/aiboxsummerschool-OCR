# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from UFUtils.utils import load_data, get_tokenizer, generate_ids
from UFUtils.spell_corrector import SpellCorrector
from UFUtils.model import Model
from enchant import DictWithPWL
from enchant.checker import SpellChecker
from copy import deepcopy
from tensorflow.python.framework import ops
import numpy as np


def correct(possible_states, text_mask):
    tokenizer, BERT_INIT_CHKPNT = get_tokenizer()
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
        total = np.sum([filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])])
        scores.append(total)

    prob_scores = np.array(scores) / np.sum(scores)
    probs = list(zip(possible_states, prob_scores))
    probs.sort(key=lambda x: x[1])

    return probs[0][0]


def main(text):
    words = load_data('../data/counts_1grams.txt')
    corrector = SpellCorrector(words)
    # modificar para todas as palavras

    my_dict = DictWithPWL("en_US", "mywords.txt")
    my_checker = SpellChecker(my_dict)

    my_checker.set_text(text)
    text_mask = deepcopy(text)
    for error in my_checker:
        err = error.word
        possible_states = corrector.edit_candidates(err)
        mask = text_mask.replace(err, '**mask**')
        correted_letter = correct(possible_states, mask)
        text_mask = deepcopy(mask.replace("**mask**", correted_letter))
    return text_mask


if __name__ == '__main__':
    text = "This is simple semple txt with erors."
    correct_text = main(text)
    print(">>>>>>>>>> BEFORE: ", text)
    print(">>>>>>>>>> AFTER: ", correct_text)

