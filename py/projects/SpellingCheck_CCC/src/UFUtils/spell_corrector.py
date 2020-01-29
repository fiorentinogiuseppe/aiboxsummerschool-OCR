# original from https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/classes/spellcorrect.py
# improved it

class SpellCorrector:
    """
    The SpellCorrector extends the functionality of the Peter Norvig's
    spell-corrector in http://norvig.com/spell-correct.html
    """

    def __init__(self, words):
        """
        :param corpus: the statistics from which corpus to use for the spell correction.
        """
        super().__init__()
        self.WORDS = words
        self.N = sum(self.WORDS.values())

    #@staticmethod
    #def tokens(text):
    #    return REGEX_TOKEN.findall(text.lower())

    def P(self, word):
        """
        Probability of `word`.
        """
        return self.WORDS[word] / self.N

    def most_probable(self, words):
        _known = self.known(words)
        if _known:
            return max(_known, key=self.P)
        else:
            return []

    @staticmethod
    def edit_step(word):
        """
        All edits that are one edit away from `word`.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """
        All edits that are two edits away from `word`.
        """
        return (e2 for e1 in self.edit_step(word)
                for e2 in self.edit_step(e1))

    def known(self, words):
        """
        The subset of `words` that appear in the dictionary of WORDS.
        """
        return set(w for w in words if w in self.WORDS)

    def edit_candidates(self, word, assume_wrong=False, fast=True):
        """
        Generate possible spelling corrections for w!pip install bert-tensorfloword.
        """

        if fast:
            ttt = self.known(self.edit_step(word)) or {word}
        else:
            ttt = self.known(self.edit_step(word)) or self.known(self.edits2(word)) or {word}

        ttt = self.known([word]) | ttt
        return list(ttt)