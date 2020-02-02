from bert import tokenization


def get_tokenizer(vocab, chkpnt):
    tokenization.validate_case_matches_checkpoint(True, chkpnt)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab, do_lower_case=True)
    return tokenizer


def generate_ids(mask, tokenizer):
    tokens = tokenizer.tokenize(mask)
    input_ids = [tokens_to_masked_ids(tokens, i, tokenizer) for i in range(len(tokens))]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, tokens_ids


def tokens_to_masked_ids(tokens, mask_ind, tokenizer):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = "[MASK]"
    masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids


def load_data(file):
    with open(file) as fopen:
        f = fopen.read().split('\n')[:-1]
    words = {}
    for l in f:
        w, c = l.split('\t')
        c = int(c)
        words[w] = c + words.get(w, 0)
    return words
