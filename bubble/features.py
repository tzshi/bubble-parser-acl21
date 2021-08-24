import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel

from .dropout import WordDropout
from .utils import BERT_TOKEN_MAPPING, from_numpy


# Embedding snippet from https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
def Embedding(num_embeddings, embedding_dim, padding_idx=0):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class CharBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharBiLSTM, self).__init__()
        print("build char sequence feature extractor: LSTM ...")
        self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim))
        )
        self.char_lstm = nn.LSTM(
            embedding_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_embeddings(input)
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class WordRep(nn.Module):
    def __init__(self, parser):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.use_char = parser._cdims > 0
        self.char_hidden_dim = 0
        if self.use_char:
            self.char_hidden_dim = parser._char_hidden
            self.char_embedding_dim = parser._cdims
            self.char_feature = CharBiLSTM(
                len(parser._charset) + 2,
                self.char_embedding_dim,
                self.char_hidden_dim,
                parser._char_dropout,
            )

        self.wdims = parser._wdims

        if self.wdims > 0:
            self.word_embedding = Embedding(len(parser._vocab) + 2, self.wdims)
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(len(parser._vocab) + 2, self.wdims)
                )
            )

        self.pdims = parser._pdims

        if self.pdims > 0:
            self.pos_embedding = Embedding(len(parser._xpos) + 2, self.pdims)

        self.drop = WordDropout(parser._word_dropout)


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
        return pretrain_emb

    def forward(
        self,
        word_inputs,
        word_seq_lengths,
        char_inputs,
        char_seq_lengths,
        char_seq_recover,
        xpos,
        emb=None,
    ):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        word_list = []

        if self.wdims > 0:
            if emb is not None:
                word_list.append(self.word_embedding(word_inputs) + emb)
            else:
                word_list.append(self.word_embedding(word_inputs))

        if self.pdims > 0:
            word_list.append(self.pos_embedding(xpos))

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(
                char_inputs, char_seq_lengths.cpu().numpy()
            )
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_list.append(char_features)

        word_embs = torch.cat(word_list, 2)

        word_represent = word_embs
        word_represent = self.drop(word_embs)

        return word_represent


class WordSequence(nn.Module):
    def __init__(self, parser):
        super(WordSequence, self).__init__()
        print("build feature extractor...")
        print("use_char: ", parser._cdims > 0)
        print("use_pos: ", parser._pdims > 0)
        print("use_bert: ", parser._bert)

        self.use_bert = parser._bert

        if self.use_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased", do_lower_case=False
            )
            self.bert_model = BertModel.from_pretrained("bert-base-cased")
            #  in_dim = self.bert_tokenizer.pooler.dense.in_features
            in_dim = 768

            if in_dim == parser._proj_dims:
                self.bert_proj = lambda x: x
            else:
                self.bert_proj = nn.Linear(in_dim, parser._bilstm_dims)
        else:
            self.use_char = parser._cdims > 0
            self.lstm_layer = parser._bilstm_layers
            self.wordrep = WordRep(parser)
            self.input_size = parser._wdims + parser._pdims
            if self.use_char:
                self.input_size += parser._char_hidden

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.

            lstm_hidden = parser._bilstm_dims // 2

            self.lstm_1 = nn.LSTM(
                self.input_size,
                lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=parser._bilstm_dropout,
            )
            self.lstm_2 = nn.LSTM(
                lstm_hidden * 2,
                lstm_hidden,
                num_layers=self.lstm_layer - 1,
                batch_first=True,
                bidirectional=True,
                dropout=parser._bilstm_dropout,
            )
            if parser._bilstm_dims == parser._proj_dims:
                self.low_level_proj = lambda x: x
                self.high_level_proj = lambda x: x
            else:
                self.low_level_proj = nn.Linear(parser._bilstm_dims, parser._proj_dims)
                self.high_level_proj = nn.Linear(parser._bilstm_dims, parser._proj_dims)

    def forward(self, batch):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_inputs = batch["word"]
        word_seq_lengths = batch["word_length"]
        char_inputs = batch["char"]
        char_seq_lengths = batch["char_length"]
        char_seq_recover = batch["char_recover"]
        xpos_inputs = batch["xpos"]
        mask = batch["mask"].transpose(1, 0)
        emb = batch.get("emb", None)
        batch_size = word_inputs.size(0)

        if self.use_bert:
            raw = batch["raw"]

            seq_max_len = len(raw[0])

            all_input_ids = np.zeros((len(raw), 2048), dtype=int)
            all_input_type_ids = np.zeros((len(raw), 2048), dtype=int)
            all_input_mask = np.zeros((len(raw), 2048), dtype=int)
            all_word_end_mask = np.zeros((len(raw), 2048), dtype=int)

            subword_max_len = 0

            for snum, sentence in enumerate(raw):
                tokens = []
                word_end_mask = []
                input_type_ids = [0]

                tokens.append("[CLS]")
                word_end_mask.append(1)

                cleaned_words = []
                for word in sentence[1:]:
                    word = BERT_TOKEN_MAPPING.get(word, word)
                    if word == "n't" and cleaned_words:
                        cleaned_words[-1] = cleaned_words[-1] + "n"
                        word = "'t"
                    cleaned_words.append(word)

                for i, word in enumerate(cleaned_words):
                    word_tokens = self.bert_tokenizer.tokenize(word)
                    if len(word_tokens) == 0:
                        word_tokens = ["."]
                    for _ in range(len(word_tokens)):
                        word_end_mask.append(0)
                    else:
                        for _ in range(len(word_tokens)):
                            input_type_ids.append(0)
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)

                tokens.append("[SEP]")
                input_type_ids.append(0)

                # pad to sequence length for every sentence
                for i in range(seq_max_len - len(sentence)):
                    word_end_mask.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(word_end_mask) + 1)

                all_input_ids[snum, : len(input_ids)] = input_ids
                all_input_type_ids[snum, : len(input_ids)] = input_type_ids
                all_input_mask[snum, : len(input_mask)] = input_mask
                all_word_end_mask[snum, : len(word_end_mask)] = word_end_mask

            all_input_ids = from_numpy(
                np.ascontiguousarray(all_input_ids[:, :subword_max_len])
            ).to(word_inputs.device)
            all_input_type_ids = from_numpy(
                np.ascontiguousarray(all_input_type_ids[:, :subword_max_len])
            ).to(word_inputs.device)
            all_input_mask = from_numpy(
                np.ascontiguousarray(all_input_mask[:, :subword_max_len])
            ).to(word_inputs.device)
            all_word_end_mask = from_numpy(
                np.ascontiguousarray(all_word_end_mask[:, :subword_max_len])
            ).to(word_inputs.device)

            last_encoder_layer, _, hidden_states = self.bert_model(
                all_input_ids, token_type_ids=all_input_type_ids, attention_mask=all_input_mask, output_hidden_states=True, return_dict=False
            )

            low_level, high_level = hidden_states[4], hidden_states[12]

            low_level = low_level.masked_select(
                all_word_end_mask.to(torch.bool).unsqueeze(-1)
            ).reshape(len(raw), seq_max_len, low_level.shape[-1])

            high_level = high_level.masked_select(
                all_word_end_mask.to(torch.bool).unsqueeze(-1)
            ).reshape(len(raw), seq_max_len, high_level.shape[-1])

            low_level = self.bert_proj(low_level)
            high_level = self.bert_proj(high_level)

        else:
            word_represent = self.wordrep(
                word_inputs,
                word_seq_lengths,
                char_inputs,
                char_seq_lengths,
                char_seq_recover,
                xpos_inputs,
                emb=emb,
            )
            ## word_embs (batch_size, seq_len, embed_size)
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths, True)
            lstm_out, _ = self.lstm_1(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            ## lstm_out (batch_size, seq_len, hidden_size)
            low_level = lstm_out

            packed_words = pack_padded_sequence(low_level, word_seq_lengths, True)
            lstm_out, _ = self.lstm_2(packed_words, None)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            ## lstm_out (batch_size, seq_len, hidden_size)
            high_level = lstm_out

            low_level = self.low_level_proj(low_level)
            high_level = self.high_level_proj(high_level)

        return low_level, high_level

    @staticmethod
    def load_data(parser, graph):
        raw = [n.word for n in graph.nodes[:]]
        norm = [n.norm for n in graph.nodes[:]]
        word = [parser._vocab.get(n.norm, 1) for n in graph.nodes[:]]
        xpos = [parser._xpos.get(n.xpos, 1) for n in graph.nodes[:]]
        char = [[parser._charset.get(ch, 1) for ch in n.word] for n in graph.nodes[:]]

        return {
            "word": word,
            "char": char,
            "norm": norm,
            "xpos": xpos,
            "raw": raw,
        }
