import json
import numpy as np

from .utils import normalize
from .utils import convert_to_bubbles


class Word:
    def __init__(self, word, xpos=None):
        self.word = word
        self.norm = normalize(word)
        self.xpos = xpos if xpos else "_"

    def clone(self):
        return Word(self.word, self.xpos)

    def __repr__(self):
        return "{}_{}".format(self.word, self.xpos)


class Sentence:
    def __init__(self, words, xposs, heads, rels, coords):
        self.nodes = [Word("*ROOT*")] + [Word(x, y) for x, y in zip(words, xposs)]
        self.heads = [-1] + list(heads)
        self.rels = ["_"] + list(rels)
        self.coords = coords

        is_projective, bubbles, bubble_heads, bubble_rels = convert_to_bubbles(self.heads, self.rels, self.coords)

        self.is_projective = is_projective
        self.bubbles = bubbles
        self.bubble_heads = bubble_heads
        self.bubble_rels = bubble_rels

        self.in_train = (is_projective and (bubbles is not None))
