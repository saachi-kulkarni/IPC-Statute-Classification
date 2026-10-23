# Master results

Test-set metrics; hyperparameters and thresholds tuned on dev only. `regex_mention` is a diagnostic (predicts sections cited verbatim in the text).

| model | family | thresholds | test_micro_f1 | test_macro_f1 | test_micro_p | test_micro_r | test_samples_f1 | test_head_f1 | test_mid_f1 | test_tail_f1 | dev_micro_f1 | dev_macro_f1 | avg_pred_labels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword | baseline | t0.5 | 0.1539 | 0.0913 | 0.1696 | 0.1409 | 0.1140 | 0.1479 | 0.0627 | 0.0617 | 0.1533 | 0.0908 | 3.2423 |
| keyword | baseline | global | 0.1756 | 0.1071 | 0.1623 | 0.1914 | 0.1308 | 0.1761 | 0.0825 | 0.0606 | 0.1741 | 0.1061 | 4.6036 |
| keyword | baseline | per_label | 0.1704 | 0.1430 | 0.1147 | 0.3311 | 0.1400 | 0.2338 | 0.1148 | 0.0778 | 0.1717 | 0.1479 | 11.2626 |
| regex_mention (diagnostic) | baseline | t0.5 | 0.7007 | 0.6271 | 0.7028 | 0.6985 | 0.6399 | 0.7592 | 0.6207 | 0.4972 | 0.7053 | 0.6277 | 3.8791 |
| regex_mention (diagnostic) | baseline | global | 0.7007 | 0.6271 | 0.7028 | 0.6985 | 0.6399 | 0.7592 | 0.6207 | 0.4972 | 0.7053 | 0.6277 | 3.8791 |
| regex_mention (diagnostic) | baseline | per_label | 0.7007 | 0.6271 | 0.7028 | 0.6985 | 0.6399 | 0.7592 | 0.6207 | 0.4972 | 0.7053 | 0.6277 | 3.8791 |
| tfidf_lr | baseline | t0.5 | 0.6468 | 0.5496 | 0.5620 | 0.7618 | 0.6238 | 0.6747 | 0.5383 | 0.4321 | 0.6449 | 0.5428 | 5.2901 |
| tfidf_lr | baseline | global | 0.6815 | 0.5812 | 0.7121 | 0.6534 | 0.6439 | 0.7058 | 0.5738 | 0.4602 | 0.6846 | 0.5774 | 3.5813 |
| tfidf_lr | baseline | per_label | 0.6908 | 0.5854 | 0.7072 | 0.6751 | 0.6545 | 0.7078 | 0.5817 | 0.4631 | 0.6995 | 0.5967 | 3.7257 |
| bert-base-uncased_head_512 | bert | t0.5 | 0.3656 | 0.2961 | 0.2384 | 0.7830 | 0.3726 | 0.4267 | 0.2539 | 0.2038 | 0.3711 | 0.2970 | 12.8168 |
| bert-base-uncased_head_512 | bert | global | 0.4895 | 0.3897 | 0.4637 | 0.5183 | 0.4728 | 0.5166 | 0.3892 | 0.2594 | 0.4948 | 0.3983 | 4.3625 |
| bert-base-uncased_head_512 | bert | per_label | 0.4970 | 0.4173 | 0.4293 | 0.5901 | 0.4819 | 0.5464 | 0.4090 | 0.2927 | 0.5144 | 0.4383 | 5.3642 |
| inlegalbert_head_512 | bert | t0.5 | 0.3939 | 0.3248 | 0.2649 | 0.7677 | 0.3948 | 0.4450 | 0.2840 | 0.2417 | 0.3985 | 0.3260 | 11.3108 |
| inlegalbert_head_512 | bert | global | 0.5036 | 0.4162 | 0.4992 | 0.5082 | 0.4795 | 0.5279 | 0.4223 | 0.2949 | 0.5103 | 0.4269 | 3.9734 |
| inlegalbert_head_512 | bert | per_label | 0.5205 | 0.4447 | 0.4662 | 0.5891 | 0.5001 | 0.5566 | 0.4381 | 0.3360 | 0.5365 | 0.4637 | 4.9326 |
| inlegalbert_head_tail_512 | bert | t0.5 | 0.3849 | 0.3147 | 0.2569 | 0.7672 | 0.3910 | 0.4348 | 0.2747 | 0.2308 | 0.3891 | 0.3161 | 11.6576 |
| inlegalbert_head_tail_512 | bert | global | 0.4960 | 0.4134 | 0.4836 | 0.5091 | 0.4730 | 0.5197 | 0.4139 | 0.3032 | 0.5001 | 0.4170 | 4.1080 |
| inlegalbert_head_tail_512 | bert | per_label | 0.5114 | 0.4357 | 0.4583 | 0.5784 | 0.4949 | 0.5504 | 0.4238 | 0.3293 | 0.5271 | 0.4544 | 4.9263 |
