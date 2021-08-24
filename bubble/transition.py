# -*- coding: utf-8 -*-

import torch
from enum import Enum

transitions = ["shift", "left", "right", "bubble_open", "bubble_close", "bubble_attach"]
Transitions = Enum('Transitions', " ".join(transitions), start=0)

class BubbleHybridOracle:

    def __init__(self, gold, features, feature_proj, ind_features, feat_num):
        if type(gold) is int:
            self.bubbles = [(i, i) for i in range(gold)]
        else:
            bubbles, bubble_heads, bubble_rels = gold
            self.gold_bubbles = bubbles
            self.gold_bubble_heads = {bubbles[i]: bubbles[bubble_heads[i]] for i in range(1, len(bubble_heads))}
            self.gold_bubble_rels = {bubbles[i]: bubble_rels[i] for i in range(1, len(bubble_heads))}

            self.gold_edges = set()
            for i in range(1, len(bubble_heads)):
                self.gold_edges.add((bubbles[bubble_heads[i]], bubbles[i]))

            self.bubbles = [x for x in bubbles if x[0] == x[1]]

        self.length = len(self.bubbles)
        self.bubble_heads = [-1 for i in self.bubbles]
        self.bubble_rels = ["_" for i in self.bubbles]
        self.stack = [0]
        self.buffer = list(reversed(range(1, len(self.bubbles))))
        self.stack_fts = [-1, -1, -1] + [0]
        self.buffer_fts = [-1] + list(reversed(range(1, len(self.bubbles))))
        self.feature_idx = [i for i in range(len(self.bubbles))]
        self.edge_idx = set()
        self.closed_bubbles = set()
        self.opened_bubbles = set()
        self.leftmost = [i for i in range(len(self.bubbles))]
        self.rightmost = [i for i in range(len(self.bubbles))]

        self.features = [x for x in features]
        self.feature_proj = feature_proj
        self.ind_features = ind_features
        self.feat_num = feat_num

    def shift(self, lbl):
        assert(self.can_shift())
        self.stack.append(self.buffer.pop())
        self.stack_fts.append(self.buffer_fts.pop())

    def can_shift(self):
        return len(self.buffer) > 0

    def left(self, lbl):
        assert(self.can_left())
        dep = self.stack.pop()
        self.stack_fts.pop()
        self.leftmost[self.buffer[-1]] = self.leftmost[dep]
        self.bubble_heads[dep] = self.buffer[-1]
        self.bubble_rels[dep] = lbl
        self.edge_idx.add((self.buffer[-1], dep))

    def can_left(self):
        return len(self.buffer) > 0 and \
            len(self.stack) > 0 and \
            self.stack[-1] != 0 and \
            (self.stack[-1] not in self.opened_bubbles) and \
            (self.buffer[-1] not in self.opened_bubbles)

    def right(self, lbl):
        assert(self.can_right())
        dep = self.stack[-1]
        self.rightmost[self.stack[-2]] = self.rightmost[dep]
        self.bubble_heads[dep] = self.stack[-2]
        self.bubble_rels[dep] = lbl
        self.edge_idx.add((self.stack[-2], dep))
        self.stack.pop()
        self.stack_fts.pop()

    def can_right(self):
        return (len(self.stack) > 1) and \
            (self.stack[-1] not in self.opened_bubbles) and \
            (self.stack[-2] not in self.opened_bubbles)

    def bubble_attach(self, lbl):
        assert(self.can_bubble_attach())
        bb, be = self.leftmost[self.stack[-2]], self.rightmost[self.stack[-1]]
        self.leftmost[self.stack[-2]], self.rightmost[self.stack[-2]] = bb, be
        self.bubbles[self.stack[-2]] = (bb, be)
        self.bubble_heads[self.stack[-1]] = self.stack[-2]
        self.bubble_rels[self.stack[-1]] = lbl
        self.edge_idx.add((self.stack[-2], self.stack[-1]))
        if lbl == "conj":
            self.features.append(self.features[self.stack_fts[-2]] + [self.features[self.stack_fts[-1]]])
            self.stack_fts[-2] = len(self.features) - 1
        self.stack.pop()
        self.stack_fts.pop()

    def can_bubble_attach(self):
        return (len(self.stack) > 1) and \
            (self.stack[-2] in self.opened_bubbles) and \
            (self.stack[-1] not in self.opened_bubbles)

    def bubble_open(self, lbl):
        assert(self.can_bubble_open())
        tmp2 = self.stack.pop()
        ft_tmp2 = self.stack_fts.pop()
        tmp1 = self.stack.pop()
        ft_tmp1 = self.stack_fts.pop()
        ll, rr = self.leftmost[tmp1], self.rightmost[tmp2]
        self.bubble_heads[tmp1] = len(self.bubbles)
        self.bubble_heads[tmp2] = len(self.bubbles)
        self.bubble_rels[tmp1] = "conj"
        self.bubble_rels[tmp2] = lbl
        self.edge_idx.add((len(self.bubbles), tmp1))
        self.edge_idx.add((len(self.bubbles), tmp2))
        self.stack.append(len(self.bubbles))
        self.opened_bubbles.add(len(self.bubbles))
        self.bubbles.append((ll, rr))
        self.bubble_heads.append(-1)
        self.bubble_rels.append("_")
        self.leftmost.append(ll)
        self.rightmost.append(rr)
        self.stack_fts.append(len(self.features))
        self.feature_idx.append(len(self.features))
        if lbl == "conj":
            self.features.append([self.features[ft_tmp1], self.features[ft_tmp2]])
        else:
            self.features.append([self.features[ft_tmp1]])

    def can_bubble_open(self):
        return (len(self.stack) > 1) and \
            (self.stack[-1] not in self.opened_bubbles) and \
            (self.stack[-2] != 0) and \
            (self.stack[-2] not in self.opened_bubbles)

    def bubble_close(self, lbl):
        assert(self.can_bubble_close())
        tmp = self.stack.pop()
        self.buffer.append(tmp)
        self.closed_bubbles.add(tmp)
        self.opened_bubbles.discard(tmp)
        self.buffer_fts.append(len(self.features))
        self.feature_idx[tmp] = len(self.features)
        self.features.append(self.feature_proj(torch.mean(torch.stack(self.features[self.stack_fts.pop()]), dim=0)))

    def can_bubble_close(self):
        return (len(self.stack) > 1) and \
            (self.stack[-1] in self.opened_bubbles)

    def get_features(self):
        features = []
        for i in range(self.feat_num):
            features.append(self.stack_fts[-i-1] if len(self.stack) > i else -1)
        features = list(reversed(features))
        features.append(self.buffer_fts[-1] if len(self.buffer) > 0 else -1)

        ret = []
        for f in features:
            if type(self.features[f]) is list:
                ret.append(self.feature_proj(torch.mean(torch.stack(self.features[f]), dim=0)))
                ret.append(self.ind_features[1])
            elif f == -1:
                ret.append(self.ind_features[3])
            else:
                ret.append(self.features[f])
                if f in self.closed_bubbles:
                    ret.append(self.ind_features[2])
                else:
                    ret.append(self.ind_features[0])
        return torch.cat(ret)

    def get_label_features(self, action):
        if action is Transitions.shift:
            return None
        elif action is Transitions.bubble_close:
            return None
        elif action is Transitions.left:
            head = self.buffer_fts[-1]
        else:
            head = self.stack_fts[-2]
        dep = self.stack_fts[-1]
        features = [head, dep]
        ret = []
        for f in features:
            if type(self.features[f]) is list:
                ret.append(self.feature_proj(torch.mean(torch.stack(self.features[f]), dim=0)))
                ret.append(self.ind_features[1])
            else:
                ret.append(self.features[f])
                if f in self.closed_bubbles:
                    ret.append(self.ind_features[2])
                else:
                    ret.append(self.ind_features[0])
        return torch.cat(ret)

    def take_action(self, action, lbl):
        if action is Transitions.shift:
            self.shift(lbl)
        elif action is Transitions.left:
            self.left(lbl)
        elif action is Transitions.right:
            self.right(lbl)
        elif action is Transitions.bubble_attach:
            self.bubble_attach(lbl)
        elif action is Transitions.bubble_open:
            self.bubble_open(lbl)
        elif action is Transitions.bubble_close:
            self.bubble_close(lbl)

    def can_take_action(self, action):
        if action is Transitions.shift:
            return self.can_shift()
        elif action is Transitions.left:
            return self.can_left()
        elif action is Transitions.right:
            return self.can_right()
        elif action is Transitions.bubble_attach:
            return self.can_bubble_attach()
        elif action is Transitions.bubble_open:
            return self.can_bubble_open()
        elif action is Transitions.bubble_close:
            return self.can_bubble_close()

    def action_masks(self):
        return [self.can_take_action(action) for action in Transitions]

    def is_done(self):
        return len(self.buffer) == 0 and len(self.stack) == 1 and len(self.opened_bubbles) == 0

    def oracle(self):
        if self.is_done():
            return True, None

        edges = {(self.bubbles[h], self.bubbles[d]) for h, d in self.edge_idx}
        has_child = False
        for h, d in self.gold_edges:
            if h == self.bubbles[self.stack[-1]] and (h, d) not in edges:
                has_child = True
                break

        dep_bubble = self.bubbles[self.stack[-1]]
        if self.stack[-1] >= self.length and self.stack[-1] in self.opened_bubbles \
            and dep_bubble in self.gold_bubble_heads and (dep_bubble, dep_bubble) not in edges:
            return Transitions.bubble_close, None

        if has_child and len(self.buffer):
            return Transitions.shift, None

        if len(self.buffer) and (self.bubbles[self.buffer[-1]], self.bubbles[self.stack[-1]]) in self.gold_edges:
            if self.stack[-1] in self.opened_bubbles:
                return Transitions.bubble_close, None
            else:
                return Transitions.left, self.gold_bubble_rels[self.bubbles[self.stack[-1]]]

        if len(self.stack) >= 2 and (self.bubbles[self.stack[-2]], self.bubbles[self.stack[-1]]) in self.gold_edges:
            if self.stack[-1] in self.opened_bubbles:
                return Transitions.bubble_close, None
            else:
                return Transitions.right, self.gold_bubble_rels[self.bubbles[self.stack[-1]]]

        if dep_bubble not in self.gold_bubble_heads:
            if len(self.buffer):
                return Transitions.shift, None
            else:
                return False, None

        head_bubble = self.gold_bubble_heads[dep_bubble]

        if len(self.stack) > 1 and self.bubbles[self.stack[-2]] in self.gold_bubble_heads \
            and self.gold_bubble_heads[self.bubbles[self.stack[-2]]] == self.gold_bubble_heads[self.bubbles[self.stack[-1]]] \
            and self.leftmost[self.stack[-2]] == head_bubble[0]:
            return Transitions.bubble_open, self.gold_bubble_rels[self.bubbles[self.stack[-1]]]

        if len(self.stack) >= 2 and self.bubbles[self.stack[-2]][0] == head_bubble[0]:
            if self.stack[-1] in self.opened_bubbles:
                return Transitions.bubble_close, None
            else:
                return Transitions.bubble_attach, self.gold_bubble_rels[self.bubbles[self.stack[-1]]]
        if len(self.buffer):
            return Transitions.shift, None

        return False, None
