import math

from itertools import combinations
from collections import defaultdict


class ReductionSummarizer():

    _stop_words = frozenset()
    #Overriding property
    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)


## Rating the sentences
def rate_sentences(self, document):
    sentences_words = [(s, self._to_words_set(s)) for s in document.sentences]
    ratings = defaultdict(float)

        # Iterating over the sentences in order to rank them
    for (sentence1, words1), (sentence2, words2) in combinations(sentences_words, 2):
        rank = self._rate_sentences_edge(words1, words2)
        ratings[sentence1] += rank
        ratings[sentence2] += rank

    return ratings

def _to_words_set(self, sentence):
    words = map(self.normalize_word, sentence.words)
    return [self.stem_word(w) for w in words if w not in self._stop_words]

def _rate_sentences_edge(self, words1, words2):
    rank = 0
    for w1 in words1:
        for w2 in words2:
            rank += int(w1 == w2)

    if rank == 0:
        return 0.0

    assert len(words1) > 0 and len(words2) > 0
    norm = math.log(len(words1)) + math.log(len(words2))
    return 0.0 if norm == 0.0 else rank / norm

#Importing necessary packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.reduction import ReductionSummarizer


def summarizer2(rawdoc):
    text = rawdoc

    # For Strings
    parser=PlaintextParser.from_string(text,Tokenizer("english"))
    # Using Reduction Summarizer
    summarizer = ReductionSummarizer()


    #Summarize the document with 4 sentences
    summary = summarizer(parser.document,3)
    doc=""
    for sentence in summary:
        doc+=(sentence)
    return doc, rawdoc, len(rawdoc.split(' ')), len(doc.split(' ')), len(rawdoc.split('.')), len(doc.split('.'))




