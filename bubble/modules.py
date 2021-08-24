from collections import defaultdict, Counter
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import BilinearMatrixAttention
from .transition import BubbleHybridOracle, Transitions
from .utils import extract_from_bubbles, extract_tree_from_bubbles


class ParserModule(ABC):
    @property
    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    @abstractmethod
    def load_data(parser, graph):
        pass

    @staticmethod
    @abstractmethod
    def batch_label(batch):
        pass

    @abstractmethod
    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        pass

    @abstractmethod
    def metrics(self, results):
        pass


class SequenceLabeler(nn.Module, ParserModule):
    def __init__(
        self, parser, layer_size, hidden_size, label_size, dropout=0.0
    ):
        super(SequenceLabeler, self).__init__()
        print("build sequence labeling network...", self.__class__.name)

        self.label_size = label_size + 2
        self.parser = parser

        lst = []
        for i in range(layer_size):
            if i == 0:
                lst.append(nn.Linear(parser._proj_dims, hidden_size))
            else:
                lst.append(nn.Linear(hidden_size, hidden_size))

            lst.append(nn.PReLU())
            lst.append(nn.Dropout(dropout))

        if layer_size > 0:
            lst.append(nn.Linear(hidden_size, self.label_size))
        else:
            lst.append(nn.Linear(parser._proj_dims, self.label_size))

        self.transform = nn.Sequential(*lst)

        self.loss = nn.NLLLoss(ignore_index=0, reduction="sum")

    def calculate_loss(self, lstm_features, batch, graphs):
        lstm_features = lstm_features[0]
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        outs = outs.view(batch_size * seq_len, -1)
        score = F.log_softmax(outs, 1)
        total_loss = self.loss(score, batch_label.view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch, graphs):
        lstm_features = lstm_features[0]
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        outs = outs.view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long() * tag_seq

        batch["pred_" + self.name] = tag_seq

        return tag_seq

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        overlaped = pred == gold
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]


class XPOSTagger(SequenceLabeler):

    name = "XPOS"

    @staticmethod
    def load_data(parser, graph):
        labels = [0] + [parser._xpos.get(n.xpos, 1) for n in graph.nodes[1:]]
        return {"xpos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["xpos"]


class UPOSTagger(SequenceLabeler):

    name = "UPOS"

    @staticmethod
    def load_data(parser, graph):
        labels = [0] + [parser._upos.get(n.upos, 1) for n in graph.nodes[1:]]
        return {"upos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["upos"]


class BubbleHybridParser(nn.Module, ParserModule):

    name = "BubbleHybrid"

    def __init__(self, parser, hidden_size, dropout=0.0, stack_fts=3, rescore=True, bubble_feature_fn="tanh"):
        super(BubbleHybridParser, self).__init__()
        print("build bubble hybrid parser...", self.__class__.name)

        self.stack_fts = stack_fts
        self.rescore = rescore
        self.bubble_feature_fn = bubble_feature_fn

        self.action_clf = nn.Sequential(
            nn.Linear((parser._proj_dims + parser._idims) * (self.stack_fts + 1), hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(Transitions)),
        )

        self.label_clf = nn.Sequential(
            nn.Linear((parser._proj_dims + parser._idims) * 2, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(parser._irels)),
        )

        self.rescore_clf = nn.Sequential(
            nn.Linear(parser._proj_dims * 3, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        if self.bubble_feature_fn == "tanh":
            self.projection= nn.Sequential(
                nn.Linear(parser._proj_dims, parser._proj_dims),
                nn.Tanh(),
            )
        else:
            self.projection = (lambda x: x)

        self.feat_atom = nn.Parameter(torch.randn(parser._idims) * 0.1)
        self.feat_open = nn.Parameter(torch.randn(parser._idims) * 0.1)
        self.feat_close = nn.Parameter(torch.randn(parser._idims) * 0.1)
        self.feat_empty = nn.Parameter(torch.randn(parser._idims + parser._proj_dims) * 0.1)
        self.ind_features = [self.feat_atom, self.feat_open, self.feat_close, self.feat_empty]

        self.a_loss = nn.NLLLoss(ignore_index=-1, reduction="sum")
        self.l_loss = nn.NLLLoss(ignore_index=-1, reduction="sum")
        self.r_loss = nn.BCEWithLogitsLoss(reduction="sum")

        self.parser = parser

    def calculate_loss(self, lstm_features, batch, graphs):
        lstm_features = lstm_features[1]
        batch_size = lstm_features.size(0)

        action_fts = []
        action_msks = []
        action_targets = []
        label_fts = []
        label_targets = []
        rescore_fts = []
        rescore_targets = []

        for g_i, graph in enumerate(graphs):
            features = [lstm_features[g_i, i] for i in range(len(graph.nodes))]
            oracle = BubbleHybridOracle((graph.bubbles, graph.bubble_heads, graph.bubble_rels), features, self.projection, self.ind_features, self.stack_fts)
            while True:
                action, lbl = oracle.oracle()

                if action is True or action is False:
                    break

                action_masks = oracle.action_masks()
                if sum(action_masks) > 1:
                    features = oracle.get_features()

                    action_fts.append(features)
                    action_msks.append([not x for x in action_masks])
                    action_targets.append(action.value)

                if lbl is not None:
                    features = oracle.get_label_features(action)

                    label_fts.append(features)
                    label_targets.append(self.parser._rels.get(lbl, 0))

                oracle.take_action(action, lbl)

            conf_bubble_idx = {b: i for i, b in enumerate(oracle.bubbles)}
            for ccp_idx in range(len(graph.nodes), len(graph.bubbles)):
                ccp = graph.bubbles[ccp_idx]
                conjs = []
                for i in range(len(graph.bubbles)):
                    if graph.bubble_heads[i] == ccp_idx and graph.bubble_rels[i] == "conj":
                        conjs.append(i)
                first_conj = conjs[0]
                last_conj = conjs[-1]

                for i in range(len(graph.bubbles)):
                    if graph.bubble_heads[i] == ccp_idx:
                        if graph.bubbles[i][1] < ccp[0]:
                            # left of ccp
                            rescore_fts.append(torch.cat([
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[i]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[first_conj]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[last_conj]]]],
                            ]))
                            rescore_targets.append(1)
                        elif  graph.bubbles[i][0] > ccp[1]:
                            # right of ccp
                            rescore_fts.append(torch.cat([
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[i]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[last_conj]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[first_conj]]]],
                            ]))
                            rescore_targets.append(1)
                if first_conj < len(graph.nodes):
                    for i in range(len(graph.bubbles)):
                        if graph.bubble_heads[i] == first_conj and graph.bubbles[i][1] < first_conj:
                            # left child of first conj
                            rescore_fts.append(torch.cat([
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[i]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[first_conj]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[last_conj]]]],
                            ]))
                            rescore_targets.append(0)
                if last_conj < len(graph.nodes):
                    for i in range(len(graph.bubbles)):
                        if graph.bubble_heads[i] == last_conj and graph.bubbles[i][0] > last_conj:
                            # left child of first conj
                            rescore_fts.append(torch.cat([
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[i]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[last_conj]]]],
                                oracle.features[oracle.feature_idx[conf_bubble_idx[graph.bubbles[first_conj]]]],
                            ]))
                            rescore_targets.append(0)

        action_fts = torch.stack(action_fts)
        action_msks = torch.Tensor(action_msks).bool().to(features.device)
        scores = self.action_clf(action_fts).masked_fill(action_msks, float("-inf"))
        scores = F.log_softmax(scores, dim=1)
        action_l = self.a_loss(scores, torch.LongTensor(action_targets).to(features.device))

        label_fts = torch.stack(label_fts)
        scores = F.log_softmax(self.label_clf(label_fts), dim=1)
        label_l = self.l_loss(scores, torch.LongTensor(label_targets).to(features.device))

        if len(rescore_targets) and self.rescore:
            rescore_fts = torch.stack(rescore_fts)
            scores = self.rescore_clf(rescore_fts)
            rescore_l = self.r_loss(scores, torch.Tensor(rescore_targets).to(features.device).unsqueeze(-1))
            loss = (action_l + label_l + rescore_l) / float(batch_size)
        else:
            loss = (action_l + label_l) / float(batch_size)

        return loss, None

    def forward(self, parser, lstm_features, batch, graphs):
        lstm_features = lstm_features[1]

        for g_i, graph in enumerate(graphs):
            features = [lstm_features[g_i, i] for i in range(len(graph.nodes))]
            conf = BubbleHybridOracle(len(graph.nodes), features, self.projection, self.ind_features, self.stack_fts)

            while not conf.is_done():
                action_masks = conf.action_masks()
                if sum(action_masks) == 0:
                    print("err in prediction")
                    print(conf.stack)
                    print(conf.buffer)
                    print(conf.open_bubbles)
                    print(conf.bubbles)
                    break

                features = conf.get_features()
                action_masks = torch.Tensor([not x for x in action_masks]).bool().to(features.device)
                scores = self.action_clf(features).masked_fill(action_masks, float("-inf")).cpu().detach().numpy()
                action, lbl = Transitions(int(np.argmax(scores))), None

                features = conf.get_label_features(action)
                if features is not None:
                    scores = self.label_clf(features).cpu().detach().numpy()
                    lbl = self.parser._irels[int(np.argmax(scores))]

                conf.take_action(action, lbl)

            if self.rescore:
                for ccp_idx in range(len(graph.nodes), len(conf.bubbles)):
                    ccp = conf.bubbles[ccp_idx]
                    conjs = []
                    last_child = 0
                    for i in range(len(conf.bubbles)):
                        if conf.bubble_heads[i] == ccp_idx and conf.bubbles[i][0] >= ccp[0] and conf.bubbles[i][1] <= ccp[1]:
                            if conf.bubble_rels[i] == "conj":
                                conjs.append(i)
                            if conf.bubbles[i][0] > conf.bubbles[last_child][0]:
                                last_child = i
                    if len(conjs) < 2:
                        continue

                    conjs = sorted(conjs, key=lambda x:conf.bubbles[x][0])
                    first_conj = conjs[0]
                    last_conj = conjs[-1]

                    rescore_fts = []
                    for i in range(len(conf.bubbles)):
                        if conf.bubble_heads[i] == ccp_idx:
                            if conf.bubbles[i][1] < ccp[0]:
                                # left of ccp
                                rescore_fts.append((i, torch.cat([
                                    conf.features[conf.feature_idx[i]],
                                    conf.features[conf.feature_idx[first_conj]],
                                    conf.features[conf.feature_idx[last_conj]],
                                ])))

                    count = 0
                    if first_conj < len(graph.nodes):
                        for i in range(len(conf.bubbles)):
                            if conf.bubble_heads[i] == first_conj and conf.bubbles[i][1] < first_conj:
                                # left child of first conj
                                rescore_fts.append((i, torch.cat([
                                    conf.features[conf.feature_idx[i]],
                                    conf.features[conf.feature_idx[first_conj]],
                                    conf.features[conf.feature_idx[last_conj]],
                                ])))
                                count += 1
                    rescore_fts = sorted(rescore_fts, key = lambda x: conf.bubbles[x[0]][0])
                    idxes = [x[0] for x in rescore_fts]
                    rescore_fts = [x[1] for x in rescore_fts]

                    if len(rescore_fts):
                        probs = abs(torch.sigmoid(self.rescore_clf(torch.stack(rescore_fts)).squeeze(-1)).cpu().detach().numpy()) + 1e-10
                        nprobs = abs(1.-probs) + 1e-10

                        log_probs = np.cumsum(np.append([0], np.log(probs)))
                        log_nprobs = np.flip(np.cumsum(np.append([0], np.flip(np.log(nprobs)))))
                        max_split = np.argmax(log_probs + log_nprobs)
                        for i in range(max_split):
                            conf.bubble_heads[idxes[i]] = ccp_idx
                        for i in range(max_split, len(idxes)):
                            conf.bubble_heads[idxes[i]] = first_conj

                    rescore_fts = []
                    for i in range(len(conf.bubbles)):
                        if conf.bubble_heads[i] == ccp_idx:
                            if conf.bubbles[i][0] > ccp[1]:
                                # right of ccp
                                rescore_fts.append((i, torch.cat([
                                    conf.features[conf.feature_idx[i]],
                                    conf.features[conf.feature_idx[last_conj]],
                                    conf.features[conf.feature_idx[first_conj]],
                                ])))
                    if last_conj < len(graph.nodes):
                        for i in range(len(conf.bubbles)):
                            if conf.bubble_heads[i] == last_conj and conf.bubbles[i][0] > last_conj:
                                # left child of first conj
                                rescore_fts.append((i, torch.cat([
                                    conf.features[conf.feature_idx[i]],
                                    conf.features[conf.feature_idx[last_conj]],
                                    conf.features[conf.feature_idx[first_conj]],
                                ])))

                    rescore_fts = sorted(rescore_fts, key = lambda x: conf.bubbles[x[0]][0])
                    idxes = [x[0] for x in rescore_fts]
                    rescore_fts = [x[1] for x in rescore_fts]

                    if len(rescore_fts):
                        probs = abs(torch.sigmoid(self.rescore_clf(torch.stack(rescore_fts)).squeeze(-1)).cpu().detach().numpy()) + 1e-10
                        nprobs = abs(1.-probs) + 1e-10

                        log_nprobs = np.cumsum(np.append([0], np.log(nprobs)))
                        log_probs = np.flip(np.cumsum(np.append([0], np.flip(np.log(probs)))))
                        max_split = np.argmax(log_probs + log_nprobs)
                        for i in range(max_split):
                            conf.bubble_heads[idxes[i]] = last_conj
                        for i in range(max_split, len(idxes)):
                            conf.bubble_heads[idxes[i]] = ccp_idx

                    left_most = [len(conf.bubbles) for i in range(len(conf.bubbles))]
                    right_most = [0 for i in range(len(conf.bubbles))]

                    for i in range(len(graph.nodes)):
                        cur = i
                        while cur != -1:
                            left_most[cur] = min(left_most[cur], i)
                            right_most[cur] = max(right_most[cur], i)
                            cur = conf.bubble_heads[cur]

                    conf.bubbles[ccp_idx] = (left_most[first_conj], right_most[last_child])

            graph.pred_bubbles = conf.bubbles
            graph.pred_bubble_heads = conf.bubble_heads
            graph.pred_bubble_rels = conf.bubble_rels
            graph.pred_heads, graph.pred_rels = extract_tree_from_bubbles(conf.bubbles, conf.bubble_heads, conf.bubble_rels)

        return None

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        tp, tp_t, fp, fn = 0, 0, 0, 0
        tp_t_whole, tp_t_inner, tp_t_outer = 0, 0, 0
        tokens, uas, las = 0, 0, 0
        if train:
            return

        for graph in graphs:
            for i in range(1, len(graph.nodes)):
                tokens += 1
                if graph.pred_heads[i] == graph.heads[i]:
                    uas += 1
                    if graph.pred_rels[i] == graph.rels[i]:
                        las += 1

            pred_coords = extract_from_bubbles(graph.pred_bubbles, graph.pred_bubble_heads, graph.pred_bubble_rels)
            graph.pred_coords = pred_coords
            gold_coords = graph.coords

            pred_dic = {x["cc"][-1][0]: tuple(sorted([tuple(y) for y in x["coord"]])) for x in pred_coords if len(x["cc"])}
            gold_dic = {x["cc"][-1][0]: tuple(sorted([tuple(y) for y in x["coord"]])) for x in gold_coords}

            for cc in gold_dic:
                if cc not in pred_dic:
                    fn += 1
            for cc in pred_dic:
                if cc not in gold_dic:
                    fp += 1
                else:
                    tp += 1
                    if pred_dic[cc] == gold_dic[cc]:
                        tp_t += 1
                    if max([x[1] for x in pred_dic[cc]]) == max([x[1] for x in gold_dic[cc]]) and min([x[0] for x in pred_dic[cc]]) == min([x[0] for x in gold_dic[cc]]):
                        tp_t_whole += 1
                    if pred_dic[cc][-2:] == gold_dic[cc][-2:]:
                        tp_t_inner += 1
                    if pred_dic[cc][0] == gold_dic[cc][0] and pred_dic[cc][-1] == gold_dic[cc][-1]:
                        tp_t_outer += 1

        results["{}-tokens".format(self.__class__.name)] += tokens
        results["{}-uas".format(self.__class__.name)] += uas
        results["{}-las".format(self.__class__.name)] += las
        results["{}-tp".format(self.__class__.name)] += tp
        results["{}-fp".format(self.__class__.name)] += fp
        results["{}-fn".format(self.__class__.name)] += fn
        results["{}-tp_t".format(self.__class__.name)] += tp_t
        results["{}-tp_t_inner".format(self.__class__.name)] += tp_t_inner
        results["{}-tp_t_outer".format(self.__class__.name)] += tp_t_outer
        results["{}-tp_t_whole".format(self.__class__.name)] += tp_t_whole


    def metrics(self, results):
        def prf(correct, precision, recall):
            f1 = 0.
            if precision != 0:
                precision = correct / precision
            if recall != 0:
                recall = correct / recall

            if precision > 0 and recall > 0:
                f1 = 2. / (1./precision + 1./recall)
            return precision * 100., recall * 100., f1 * 100.

        tp = results["{}-tp".format(self.__class__.name)]
        fp = results["{}-fp".format(self.__class__.name)]
        fn = results["{}-fn".format(self.__class__.name)]
        tp_t = results["{}-tp_t".format(self.__class__.name)]
        tp_t_inner = results["{}-tp_t_inner".format(self.__class__.name)]
        tp_t_outer = results["{}-tp_t_outer".format(self.__class__.name)]
        tp_t_whole = results["{}-tp_t_whole".format(self.__class__.name)]
        del results["{}-tp".format(self.__class__.name)]
        del results["{}-fp".format(self.__class__.name)]
        del results["{}-fn".format(self.__class__.name)]
        del results["{}-tp_t".format(self.__class__.name)]
        del results["{}-tp_t_inner".format(self.__class__.name)]
        del results["{}-tp_t_outer".format(self.__class__.name)]
        del results["{}-tp_t_whole".format(self.__class__.name)]

        pp = tp + fp
        rr = tp + fn
        precision, recall, f1 = prf(tp_t, pp, rr)

        results["metrics/{}-p".format(self.__class__.name)] = precision
        results["metrics/{}-r".format(self.__class__.name)] = recall
        results["metrics/{}-f".format(self.__class__.name)] = f1

        precision, recall, f1 = prf(tp_t_inner, pp, rr)
        results["metrics/{}-inner-p".format(self.__class__.name)] = precision
        results["metrics/{}-inner-r".format(self.__class__.name)] = recall
        results["metrics/{}-inner-f".format(self.__class__.name)] = f1

        precision, recall, f1 = prf(tp_t_outer, pp, rr)
        results["metrics/{}-outer-p".format(self.__class__.name)] = precision
        results["metrics/{}-outer-r".format(self.__class__.name)] = recall
        results["metrics/{}-outer-f".format(self.__class__.name)] = f1

        precision, recall, f1 = prf(tp_t_whole, pp, rr)
        results["metrics/{}-whole-p".format(self.__class__.name)] = precision
        results["metrics/{}-whole-r".format(self.__class__.name)] = recall
        results["metrics/{}-whole-f".format(self.__class__.name)] = f1

        tokens = results["{}-tokens".format(self.__class__.name)]
        uas = results["{}-uas".format(self.__class__.name)]
        las = results["{}-las".format(self.__class__.name)]
        uas = uas / tokens * 100. if tokens else 0.
        las = las / tokens * 100. if tokens else 0.

        results["metrics/{}-uas".format(self.__class__.name)] = uas
        results["metrics/{}-las".format(self.__class__.name)] = las
        del results["{}-tokens".format(self.__class__.name)]
        del results["{}-uas".format(self.__class__.name)]
        del results["{}-las".format(self.__class__.name)]

    @staticmethod
    def load_data(parser, graph):
        return {}

    @staticmethod
    def batch_label(batch):
        return None
