# -*- coding: utf-8 -*-

import json

from bubble.parser import BubbleParser
from bubble.data import DataProcessor

split = 1

parser = BubbleParser()

parser.load_model("./models/ptb_bert/model", batch_size=16, gpu=False)
# parser.load_model(f"./models/genia_bert_{split}/model", batch_size=16, gpu=False)
parser.load_embeddings("./embeddings/glove.6b.100")

parser._model.eval()

graphs = DataProcessor("./data/ptb-dev.ext", parser, parser._model)
# graphs = DataProcessor(f"./data/genia/split_{split}", parser, parser._model)

parser.evaluate(graphs)

output = []

# for graph in graphs.graphs:
    # output.append({
        # "words": [x.word for x in graph.nodes],
        # "heads": graph.heads,
        # "rels": graph.rels,
        # "bubbles": graph.bubbles,
        # "bubble_heads": graph.bubble_heads,
        # "bubble_rels": graph.bubble_rels,
        # "coords": graph.coords,
        # "pred_bubbles": graph.pred_bubbles,
        # "pred_bubble_heads": graph.pred_bubble_heads,
        # "pred_bubble_rels": graph.pred_bubble_rels,
        # "pred_coords": graph.pred_coords,
    # })

# with open(f"./pred.json", "w") as f:
    # json.dump(output, f)
