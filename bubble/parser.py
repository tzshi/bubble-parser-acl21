import json
import sys
import os
import time
from collections import defaultdict

import fire
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .modules import BubbleHybridParser, XPOSTagger
from .features import WordSequence
from .io import read_conll
from .utils import buildVocab
from .data import DataProcessor, DataCollate, InfiniteDataLoader
from .adamw import AdamW


class BubbleParser:
    def __init__(self, **kwargs):
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._gpu = kwargs.get("gpu", True)

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._weight_decay = kwargs.get("weight_decay", 0.0)
        self._warmup = kwargs.get("warmup", 0)

        self._clip = kwargs.get("clip", 5.0)

        self._batch_size = kwargs.get("batch_size", 16)

        self._wdims = kwargs.get("wdims", 128)
        self._edims = kwargs.get("edims", 0)
        self._cdims = kwargs.get("cdims", 32)
        self._pdims = kwargs.get("pdims", 0)
        self._idims = kwargs.get("idims", 16)

        self._word_dropout = kwargs.get("word_dropout", 0.0)

        self._char_hidden = kwargs.get("char_hidden", 128)
        self._char_dropout = kwargs.get("char_dropout", 0.0)
        self._bilstm_dims = kwargs.get("bilstm_dims", 256)
        self._bilstm_layers = kwargs.get("bilstm_layers", 2)
        self._bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        self._proj_dims = kwargs.get("proj_dims", 256)

        self._parser_dims = kwargs.get("parser_dims", 200)
        self._parser_dropout = kwargs.get("parser_dropout", 0.0)

        self._tagger_dims = kwargs.get("tagger_dims", 200)
        self._tagger_dropout = kwargs.get("tagger_dropout", 0.0)

        self._stack_fts = kwargs.get("stack_fts", 3)
        self._rescore = kwargs.get("rescore", False)
        self._bubble_feature_fn = kwargs.get("bubble_feature_fn", "tanh")

        self._bert = kwargs.get("bert", False)
        if self._bert:
            self._learning_rate = 1e-5
            self._proj_dims = 768

        self.init_model()

        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._xpos = {p: i + 2 for i, p in enumerate(vocab["xpos"])}
        self._vocab = {w: i + 2 for i, w in enumerate(vocab["vocab"])}
        self._charset = {c: i + 2 for i, c in enumerate(vocab["charset"])}
        self._wordfreq = vocab["wordfreq"]
        self._charfreq = vocab["charfreq"]

        self._irels = ["unk"] + vocab["rels"]
        self._rels = {w: i for i, w in enumerate(self._irels)}

    def load_vocab(self, filename):
        with open(filename, "rb") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "wb") as f:
            f.write(json.dumps(self._fullvocab).encode("utf-8"))
        return self

    def build_vocab(self, filename, cutoff=1):
        graphs = read_conll(filename)

        self._fullvocab = buildVocab(graphs, cutoff)
        self._load_vocab(self._fullvocab)

        return self

    def load_embeddings(self, filename):
        with open(filename + ".vocab", "rb") as f:
            _external_mappings = json.load(f)
        with open(filename + ".npy", "rb") as f:
            _external_embeddings = np.load(f)

        count = 0
        for w in self._vocab:
            if w in _external_mappings:
                count += 1
        print(
            "Loaded embeddings from", filename, count, "hits out of", len(self._vocab)
        )
        self._external_mappings = _external_mappings
        self._external_embeddings = _external_embeddings

        return self

    def save_model(self, filename):
        print("Saving model to", filename)
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "wb") as f:
            f.write(json.dumps(self._args).encode("utf-8"))
        with open(filename + ".model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_model(self, filename, **kwargs):
        print("Loading model from", filename)
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "rb") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        with open(filename + ".model", "rb") as f:
            if kwargs.get("gpu", False):
                self._model.load_state_dict(torch.load(f))
            else:
                self._model.load_state_dict(torch.load(f, map_location="cpu"))
        return self

    def init_model(self):
        self._seqrep = WordSequence(self)

        self._parser = BubbleHybridParser(
            self, self._parser_dims, self._parser_dropout
        )
        self._parser.l_weight = 1.0

        self._modules = [self._parser]

        if self._pdims == 0:
            print("including a stacked tagger")
            self._tagger = XPOSTagger(self, 1, self._tagger_dims, len(self._xpos), self._tagger_dropout)
            self._tagger.l_weight = 0.1
            self._modules.append(self._tagger)

        modules = [self._seqrep] + self._modules
        self._model = nn.ModuleList(modules)

        if self._gpu:
            print("Detected", torch.cuda.device_count(), "GPUs")
            self._device_ids = [i for i in range(torch.cuda.device_count())]
            self._model.cuda()

        return self

    def train(
        self,
        filename,
        eval_steps=100,
        decay_evals=5,
        decay_times=0,
        decay_ratio=0.5,
        dev=None,
        save_prefix=None,
        **kwargs
    ):
        train_graphs = DataProcessor(filename, self, self._model, train=True)
        train_loader = InfiniteDataLoader(
            train_graphs,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=DataCollate(self, train=True),
        )
        dev_graphs = DataProcessor(dev, self, self._model)

        optimizer = AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            betas=(self._beta1, self._beta2),
            eps=self._epsilon,
            weight_decay=self._weight_decay,
            amsgrad=False,
            warmup=self._warmup,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "max",
            factor=decay_ratio,
            patience=decay_evals,
            verbose=True,
            cooldown=1,
        )

        print("Model")
        for param_tensor in self._model.state_dict():
            print(param_tensor, "\t", self._model.state_dict()[param_tensor].size())
        print("Opt")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        t0 = time.time()
        results, eloss = defaultdict(float), 0.0
        max_dev = 0.0

        for batch_i, batch in enumerate(train_loader):
            graphs = [train_graphs.graphs[idx] for idx in batch["graphidx"]]
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw" and "_length" not in k:
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            self._model.train()
            self._model.zero_grad()

            loss = []

            seq_features = self._seqrep(batch)

            for module in self._modules:

                l, pred = module.calculate_loss(seq_features, batch, graphs)

                batch_label = module.batch_label(batch)

                if l is not None:
                    loss.append(l * module.l_weight)
                    module.evaluate(
                        results, self, graphs, pred, batch_label, mask, train=True
                    )

            loss = sum(loss)
            eloss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
            optimizer.step()

            if batch_i and batch_i % 100 == 0:
                for module in self._modules:
                    module.metrics(results)
                results["loss/loss"] = eloss
                print(batch_i // 100, "{:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                results, eloss = defaultdict(float), 0.0
                t0 = time.time()

            if batch_i and (batch_i % eval_steps == 0):
                self.save_model("{}model_latest".format(save_prefix))
                results = self.evaluate(dev_graphs)

                performance = results["metrics/BubbleHybrid-f"]

                results = defaultdict(float)
                scheduler.step(performance)
                if scheduler.in_cooldown:
                    optimizer.state = defaultdict(dict)
                    if decay_times <= 0:
                        break
                    else:
                        decay_times -= 1

                print()
                print(performance)
                print()
                if performance >= max_dev:
                    max_dev = performance
                    if save_prefix:
                        self.save_model("{}model".format(save_prefix))

        return self

    def evaluate(self, data, output_file=None):
        results = defaultdict(float)
        pred_scores = []
        pred_results = []
        gold_results = []
        self._model.eval()
        batch_size = self._batch_size
        start_time = time.time()
        train_num = len(data)

        dev_loader = DataLoader(
            data,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=DataCollate(self, train=False),
        )

        for batch in tqdm(dev_loader):
            graphs = [data.graphs[idx] for idx in batch["graphidx"]]
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw" and "_length" not in k:
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            seq_features = self._seqrep(batch)

            for module in self._modules:
                batch_label = module.batch_label(batch)
                pred = module(self, seq_features, batch, graphs)
                module.evaluate(
                    results, self, graphs, pred, batch_label, mask, train=False
                )

        decode_time = time.time() - start_time
        results["speed/speed"] = len(data) / decode_time

        for module in self._modules:
            module.metrics(results)

        print(results)

        if output_file:
            write_conll(output_file, data.sents)

        return results

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    fire.Fire(BubbleParser)
