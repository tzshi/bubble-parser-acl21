import re
import random
import os
from collections import Counter

import torch

if torch.cuda.is_available():
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).cuda()
else:
    from torch import from_numpy


BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
}


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def buildVocab(graphs, cutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    xposCount = Counter()
    relsCount = Counter()
    intrainCount = 0

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
        xposCount.update([node.xpos for node in graph.nodes])
        relsCount.update(graph.rels[1:])
        intrainCount += int(graph.in_train)
    print(f"Well-formed bubbles: {intrainCount}/{len(graphs)}")

    print("Number of tokens in training corpora: {}".format(sum(wordsCount.values())))
    print("Vocab containing {} types before cutting off".format(len(wordsCount)))
    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})

    print(
        "Vocab containing {} types, covering {} words".format(
            len(wordsCount), sum(wordsCount.values())
        )
    )
    print("Charset containing {} chars".format(len(charsCount)))
    print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rel set containing {} tags".format(len(relsCount)), relsCount)


    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "xpos": list(xposCount.keys()),
        "rels": list(relsCount.keys()),
    }

    return ret


def longest_common_suffix(list_of_strings):
    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]
    return lcs


def get_path(heads, i):
    ret = [i]
    while i > 0:
        i = heads[i]
        ret.append(i)
    return ret

def is_constituent(heads, start, end):
    if start == end:
        return True
    lmost = [1000 for i in heads]
    rmost = [-1 for i in heads]
    for i in range(1, len(heads)):
        cur = i
        while cur > 0:
            lmost[cur] = min(lmost[cur], i)
            rmost[cur] = max(rmost[cur], i)
            cur = heads[cur]

    paths = []
    for i in range(start, end+1):
        path = get_path(heads, i)
        paths.append(path)

    lcs = longest_common_suffix(paths)
    common_root_length = len(lcs)
    common_root = lcs[0]

    children = set([p[-common_root_length - 1] for p in paths if len(p) > common_root_length])
    for c in children:
        if lmost[c] < start or rmost[c] > end:
            return False

    return True

def find_subroot(heads, start, end):
    paths = []
    for i in range(start, end+1):
        path = get_path(heads, i)
        paths.append(path)

    lcs = longest_common_suffix(paths)
    common_root_length = len(lcs)
    common_root = lcs[0]
    return common_root

def extract_from_bubbles(bubbles, bubble_heads, bubble_rels):
    left_most = [len(bubbles) for i in range(len(bubbles))]
    right_most = [0 for i in range(len(bubbles))]

    for i in range(len(bubble_heads)):
        left, right = bubbles[i]

        cur = i
        while cur != -1:
            left_most[cur] = min(left_most[cur], left)
            right_most[cur] = max(right_most[cur], right)
            cur = bubble_heads[cur]

    ret = []
    for b_i, (b, e) in enumerate(bubbles):
        if b != e:
            ccp = (b, e)
            coords = []
            ccs = []
            for i in range(len(bubble_heads)):
                if bubble_heads[i] == b_i and bubble_rels[i] == "conj":
                    coords.append((left_most[i], right_most[i]))
            for i in range(len(bubble_heads)):
                if bubble_heads[i] == b_i and bubble_rels[i] == "cc":
                    ccs.append((left_most[i], right_most[i]))
            if len(coords):
                ccp = (min([x[0] for x in coords]), max([x[1] for x in coords]))
                ret.append({
                    "ccp": ccp,
                    "cc": ccs,
                    "coord": coords,
                })
    return ret

def extract_tree_from_bubbles(bubbles, bubble_heads, bubble_rels):
    left_most = [len(bubbles) for i in range(len(bubbles))]
    right_most = [0 for i in range(len(bubbles))]

    for i in range(len(bubble_heads)):
        left, right = bubbles[i]

        cur = i
        while cur != -1:
            left_most[cur] = min(left_most[cur], left)
            right_most[cur] = max(right_most[cur], right)
            cur = bubble_heads[cur]

    length = sum([b == e for b, e in bubbles])
    heads = [i for i in bubble_heads]
    rels = [r for r in bubble_rels]
    for b_i, (b, e) in enumerate(bubbles):
        if b != e:
            children = []
            outside_children = []
            for i, h in enumerate(heads):
                if h == b_i and left_most[i] >= b and right_most[i] <= e:
                    children.append(i)
                elif h == b_i:
                    outside_children.append(i)
            children = sorted(children, key=lambda x: left_most[x])
            c_0 = children[0]
            for i in range(1, len(children)):
                c_i = children[i]
                if rels[c_i] == "cc":
                    j = i
                    while j < len(children):
                        if rels[children[j]] == "conj":
                            break
                        j += 1
                    if j == len(children):
                        j = i
                        while j >= 0:
                            if rels[children[j]] == "conj":
                                break
                            j -= 1
                    heads[c_i] = children[j]
                else:
                    heads[c_i] = c_0
            for i in outside_children:
                heads[i] = c_0
            rels[c_0] = rels[b_i]
            heads[c_0] = heads[b_i]

    return heads[:length], rels[:length]

def is_projective(bubbles, bubble_heads):
    # examine bubbles do not cross
    for i in range(len(bubbles)):
        ib, ie = bubbles[i]
        for j in range(i + 1, len(bubbles)):
            jb, je = bubbles[j]
            if ib < jb <= ie < je:
                return False
            if jb < ib <= je < ie:
                return False

    for i in range(1, len(bubble_heads)):
        ib, ie = bubbles[i]
        hib, hie = bubbles[bubble_heads[i]]

        # upward edge
        if hib >= ib and hie <= ie:
            return False
        # downward edge
        elif hib <= ib and hie >= ie:
            # not bursing any bubble
            for j in range(len(bubbles)):
                if j == i or j == bubble_heads[i]:
                    continue
                jb, je = bubbles[j]
                if hib <= jb <= ib <= ie <= je <= hie:
                    return False
        # outgoing edge
        else:
            if ib < hib:
                farleft, left, right, farright = ib, ie, hib, hie
            else:
                farleft, left, right, farright = hib, hie, ib, ie
            # not bursting any bubble
            for j in range(len(bubbles)):
                if j == i or j == bubble_heads[i]:
                    continue
                jb, je = bubbles[j]
                if jb <= farleft <= left <= je < right:
                    return False
                if left < jb <= right <= farright <= je:
                    return False

            # not crossing any edge
            for j in range(1, len(bubble_heads)):
                jb, je = bubbles[j]
                hjb, hje = bubbles[bubble_heads[j]]
                # other edge is downward
                if hjb <= jb and hje >= je:
                    if hjb <= left < jb <= je < right <= hje:
                        return False
                # other edge is outgoing
                else:
                    if jb < hjb:
                        left2, right2 = je, hjb
                    else:
                        left2, right2 = hje, jb
                    if left < left2 < right < right2 or left2 < left < right2 < right:
                        return False

    return True

def fix_bursting_bubbles(bubbles, bubble_heads):
    for i in range(1, len(bubble_heads)):
        ib, ie = bubbles[i]
        hib, hie = bubbles[bubble_heads[i]]

        if hib >= ib and hie <= ie:
            continue
        elif hib <= ib and hie >= ie:
            continue
        else:
            if ib < hib:
                for j in range(len(bubbles)):
                    jb, je = bubbles[j]
                    if ie < jb <= hib <= hie <= je:
                        bubble_heads[i] = j
                        hib, hie = jb, je
            else:
                for j in range(len(bubbles)):
                    jb, je = bubbles[j]
                    if jb <= hib <= hie <= je < ib:
                        bubble_heads[i] = j
                        hib, hie = jb, je

def convert_to_bubbles(heads, rels, coords):
    bubble_heads = [i for i in heads]
    bubble_rels = [i for i in rels]
    for i in range(len(bubble_rels)):
        if bubble_rels[i] == "conj":
            bubble_rels[i] = "dep"
    bubbles = [(i, i) for i in range(len(heads))]

    if len(coords) == 0:
        return is_projective(bubbles, bubble_heads), bubbles, bubble_heads, bubble_rels

    flag = True
    for coord in coords:
        if not is_constituent(heads, coord['ccp'][0], coord['ccp'][1]):
            flag = False

        for b, e in coord['coord']:
            if not is_constituent(heads, b, e):
                flag = False

    for coord in coords:
        ccp_root = find_subroot(heads, coord['ccp'][0], coord['ccp'][1])

        for b, e in coord['coord']:
            conj_root = find_subroot(heads, b, e)
            if conj_root < b or conj_root > e:
                flag = False

    if flag is False:
        return is_projective(bubbles, bubble_heads), None, None, None

    for coord in coords:
        ccp_b, ccp_e = coord['ccp'][0], coord['ccp'][1]
        ccp_root = find_subroot(heads, ccp_b, ccp_e)

        if ccp_root < coord['ccp'][0] or ccp_root > coord['ccp'][1]:
            ccp_head = ccp_root
            rel_sets = set()
            for i in range(ccp_b, ccp_e + 1):
                if heads[i] == ccp_root:
                    rel_sets.add(rels[i])
            rel_sets = rel_sets - {'cc', 'conj', 'punct'}
            if len(rel_sets) == 0:
                ccp_rel = "dep"
            else:
                ccp_rel = list(rel_sets)[0]
        else:
            ccp_head = bubble_heads[ccp_root]
            ccp_rel = bubble_rels[ccp_root]

        bubble_heads.append(ccp_head)
        bubble_rels.append(ccp_rel)
        bubbles.append((ccp_b, ccp_e))

        inside_nodes = set(range(coord['ccp'][0], coord['ccp'][1] + 1))

        conn_nodes = set()

        for b, e in coord['coord']:
            for i in range(b, e + 1):
                inside_nodes.discard(i)
            conj_root = find_subroot(heads, b, e)
            bubble_heads[conj_root] = len(bubble_heads) - 1
            bubble_rels[conj_root] = 'conj'

        inside_nodes_2 = set()
        for i in inside_nodes:
            if heads[i] not in inside_nodes:
                inside_nodes_2.add(i)
        inside_nodes = inside_nodes_2

        for i in inside_nodes:
            if i == ccp_root:
                if rels[i] == "advmod":
                    bubble_rels[i] = "cc"
            bubble_heads[i] = len(bubble_heads) - 1

    fix_bursting_bubbles(bubbles, bubble_heads)

    return is_projective(bubbles, bubble_heads), bubbles, bubble_heads, bubble_rels
