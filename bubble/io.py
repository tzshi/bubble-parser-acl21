import os.path
import json

from tqdm import tqdm
from nltk.corpus.reader import BracketParseCorpusReader

from .const import (
    ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
)
from .graph import Sentence


def add_indices_to_terminals(tree):
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location[:-1]]
        non_terminal[0] = non_terminal[0] + "_" + str(idx + 1)
    return tree

def extract_dependency(dtree):
    heads = [int(x[HEAD]) for x in dtree]
    xposs = [x[XPOS] for x in dtree]
    words = [x[FORM] for x in dtree]
    rels = [x[DEPREL].split(":")[0] for x in dtree]
    return words, xposs, heads, rels

def extract_coords(sent, boundary=True):
    ret = []
    for sub in sent.subtrees():
        if "-CCP" in sub.label():
            ptype = sub.label().split("-")[0]
            ccp_leaves = sub.leaves()
            ccp_b, ccp_e = int(ccp_leaves[0].split("_")[-1]), int(ccp_leaves[-1].split("_")[-1])
            coords = []
            ccs = []
            shareds = []
            conns = []
            marks = []
            for n, child in enumerate(sub):
                if isinstance(child, str):
                    continue
                if "-COORD" in child.label():
                    coord_leaves = child.leaves()
                    coord_b, coord_e = int(coord_leaves[0].split("_")[-1]), int(coord_leaves[-1].split("_")[-1])
                    coords.append((coord_b, coord_e))
                    continue
                if "-CC" in child.label().replace("-CCP", ""):
                    cc_leaves = child.leaves()
                    cc_b, cc_e = int(cc_leaves[0].split("_")[-1]), int(cc_leaves[-1].split("_")[-1])
                    ccs.append((cc_b, cc_e))
                    continue
                if "-SHARED" in child.label():
                    shared_leaves = child.leaves()
                    shared_b, shared_e = int(shared_leaves[0].split("_")[-1]), int(shared_leaves[-1].split("_")[-1])
                    shareds.append((shared_b, shared_e))
                    continue
                if "-CONN" in child.label():
                    conn_leaves = child.leaves()
                    conn_b, conn_e = int(conn_leaves[0].split("_")[-1]), int(conn_leaves[-1].split("_")[-1])
                    conns.append((conn_b, conn_e))
                    continue
                if "-MARK" in child.label():
                    mark_leaves = child.leaves()
                    mark_b, mark_e = int(mark_leaves[0].split("_")[-1]), int(mark_leaves[-1].split("_")[-1])
                    marks.append((mark_b, mark_e))
                    continue

            if boundary:
                ccp_b = min([x[0] for x in coords])
                ccp_e = max([x[1] for x in coords])
            ret.append({
                "ccp": (ccp_b, ccp_e),
                "coord": coords,
                "cc": ccs,
                "shared": shareds,
                "conn": conns,
                "mark": marks,
                "ptype": ptype,
            })

    return ret

def read_conll(filename):
    dirname, basename = os.path.split(filename)

    graphs = []
    with open(filename + ".conllu") as f:
        dcorpus = f.read().strip().split("\n\n")
        dcorpus = [[y.split("\t") for y in x.split("\n")] for x in dcorpus]

    if os.path.isfile(filename + ".json"):
        with open(filename + ".json") as f:
            all_coords = json.load(f)

        for coords, dtree in tqdm(zip(all_coords, dcorpus)):
            words, xposs, heads, rels = extract_dependency(dtree)
            for i in range(len(words) - 1):
                if words[i] in {"and", "but"} and words[i + 1] in {"not", "rather"}:
                    if (heads[i] == i+2):
                        heads[i] = heads[heads[heads[i+1] - 1] - 1]
            graphs.append(Sentence(words, xposs, heads, rels, coords))
    else:
        corpus = BracketParseCorpusReader(dirname, basename + ".cleaned").parsed_sents()
        corpus = [add_indices_to_terminals(x) for x in tqdm(corpus)]

        # making sure these are referring to the same trees
        for ctree, dtree in tqdm(zip(corpus, dcorpus)):
            assert(len(ctree.leaves()) == len(dtree))

            words, xposs, heads, rels = extract_dependency(dtree)
            coords = extract_coords(ctree)
            graphs.append(Sentence(words, xposs, heads, rels, coords))

    return graphs
