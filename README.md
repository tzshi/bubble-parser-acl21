## About this repo

This repo hosts code for "Transition-based Bubble Parsing: Improvements on Coordination Structure Prediction" (Shi and Lee, ACL 2021).

## Running the code

See scripts under `scripts/` for example training scripts, and see `test.py` for example inferencing script.

Some available parameters:
|Parameter|Description|
|---|---|
|`wdims`|Word embedding dimension|
|`cdims`|Character embedding dimesion|
|`edims`|External word embedding (e.g., GloVe) dimension|
|`pdims`|POS tag embedding dimension|
|`idims`|Indicator embedding dimension|
|`bilstm-dims`|Bi-LSTM hidden dimension|
|`bilstm-layers`|Bi-LSTM layers|
|`bilstm-dropout`|Bi-LSTM dropout|
|`char-hidden`|Char LSTM dimension (always 1 layer)|
|`char-dropout`|Char LSTM dropout|
|`parser-dims`|Parser MLP hidden dimension|
|`parser-dropout`|Parser MLP dropout|
|`stack-fts`|Number of positional features taken from stack|
|`rescore`|`True` or `False`, enabling/disabling the rescoring module|
|`bert`|`True` or `False`, using pre-trained BERT features or not|



## Reference

```
@InProceedings{shi-lee-21-transition,
    title = "Transition-based Bubble Parsing: Improvements on Coordination Structure Prediction"
    author = "Shi, Tianze  and
              Lee, Lillian",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = "2021",
    address = "Online",
    pages = "7167--7182",
    publisher = "Association for Computational Linguistics",
}
```
