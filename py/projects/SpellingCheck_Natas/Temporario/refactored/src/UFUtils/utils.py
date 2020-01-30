import codecs
import re
from difflib import SequenceMatcher
import spacy
import nltk
import torch
from enchant.checker import SpellChecker
from transformers import AlbertModel, AlbertTokenizer

def load_text(file_path):
    text = []
    with codecs.open(file_path, encoding="utf-8-sig") as f:
        for line in f:
            text.append(line)
    return ' '.join(text)


def get_personslist(text):
    personslist = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0, (chunk.leaves()[0][0]))
    return list(set(personslist))


def predict_word(text_original, predictions, maskids, tokenizer, suggestedwords):
    pred_words=[]
    nlp = spacy.load('en_core_web_md')
    for i in range(len(maskids)):
        # indice 1 e o index
        preds = torch.topk(predictions[0, maskids[i]], k=predictions.shape[2])[1]
        #preds = torch.topk(predictions[0, maskids[i]], k=200)[1]
        indices = preds.cpu().numpy()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        #print(list1)
        list2 = suggestedwords[i]
        #print(list2)
        simmax=0
        predicted_token=''
        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, re.sub('[^a-zA-Z0-9]+', '', word1), word2.lower()).ratio()
                if s is not None and s >= simmax:
                    simmax = s
                    predicted_token = word1
        text_original = text_original.replace('[MASK]', '<mark>'+re.sub('[^a-zA-Z0-9]+', '', predicted_token)+'</mark>', 1)
    return text_original


def correct_spell(text):
    rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ',
            '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ',
            '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ',
            '(': ' ( ', ')': ' ) ', "s'": "s '"}

    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text_original = str(text)
    personslist = get_personslist(text)
    ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']

    # using enchant.checker.SpellChecker, identify incorrect words
    d = SpellChecker("en_US")
    words = text.split()
    incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]

    # using enchant.checker.SpellChecker, get suggested replacements
    #suggestedwords = [d.suggest(w) for w in incorrectwords]
    suggestedwords =[]
    for w in incorrectwords:
        sugs = d.suggest(w)
        suggestedwords_tmp = []
        for sug in sugs:
            suggestedwords_tmp.append(re.sub('[^a-zA-Z0-9]+', '', sug))
        suggestedwords.append(suggestedwords_tmp)

    for w in incorrectwords:
        text = text.replace(w, '[MASK]')
        text_original = text_original.replace(w, '[MASK]')

    # Load, train and predict using pre-trained model
    model_name = "albert-xlarge-v2"
    tokenizer = AlbertTokenizer.from_pretrained(model_name, cache_dir="./model/{}/".format(model_name))
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']

    # Create the segments tensors
    segs = [i for i, e in enumerate(tokenized_text) if e == "."]
    segments_ids=[]
    prev=-1

    for k, s in enumerate(segs):
        segments_ids = segments_ids + [k] * (s-prev)
        prev=s

    segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
    segments_tensors = torch.tensor([segments_ids])
    segments_tensors = segments_tensors.to('cuda')

    # prepare Torch inputs
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')


    # Load pre-trained model
    model = AlbertModel.from_pretrained(model_name, cache_dir="../model/{}/".format(model_name))
    model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # The last hidden-state is the first element of the output tuple
        predictions = outputs[0]

    text_original = predict_word(text_original, predictions, MASKIDS, tokenizer, suggestedwords)
    return text_original
