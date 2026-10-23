| Model | Thresholds | Micro-F1 | Macro-F1 | Micro-P | Micro-R | Samples-F1 | Head F1 | Mid F1 | Tail F1 | Dev micro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR | per_label | 0.691 | 0.585 | 0.707 | 0.675 | 0.654 | 0.708 | 0.582 | 0.463 | 0.699 |
| InLegalBERT (head) | per_label | 0.520 | 0.445 | 0.466 | 0.589 | 0.500 | 0.557 | 0.438 | 0.336 | 0.537 |
| Citation diagnostic | t0.5 | 0.701 | 0.627 | 0.703 | 0.699 | 0.640 | 0.759 | 0.621 | 0.497 | 0.705 |
| BERT-base (head) | per_label | 0.497 | 0.417 | 0.429 | 0.590 | 0.482 | 0.546 | 0.409 | 0.293 | 0.514 |
| InLegalBERT (head+tail) | per_label | 0.511 | 0.436 | 0.458 | 0.578 | 0.495 | 0.550 | 0.424 | 0.329 | 0.527 |
| Keyword | global | 0.176 | 0.107 | 0.162 | 0.191 | 0.131 | 0.176 | 0.083 | 0.061 | 0.174 |
