import numpy as np
import json
from .io import read_conll
from .utils import buildVocab

files = [
    "/path/to/data/ptb-train.ext",
    "/path/to/data/ptb-dev.ext",
    "/path/to/data/ptb-test.ext",
    "/path/to/data/genia/train_1",
    "/path/to/data/genia/test_1",
]

embed_dim = 100

embedding_file = "/path/to/embeddings/glove.6B.{}d.txt".format(embed_dim)

output_file = "./embeddings/glove.6b.{}".format(embed_dim)

sents = []

for filename in files:
    sents.extend(read_conll(filename))

graphs = sents


vocab = buildVocab(graphs, cutoff=0)

fvocab = set(vocab["vocab"])


evocab = {}
evecs = [[0.0 for i in range(embed_dim)]]
cur = 1

with open(embedding_file, "r") as f:
    for l in f:
        line = l.split()
        assert len(line) == embed_dim + 1
        word = line[0]
        if word in fvocab and word not in evocab:
            evocab[word] = cur
            vec = [float(x) for x in line[1:]]
            evecs.append(vec)
            cur += 1

with open(output_file + ".vocab", "wb") as f:
    f.write(json.dumps(evocab).encode("utf-8"))

np.save(output_file + ".npy", np.array(evecs))
