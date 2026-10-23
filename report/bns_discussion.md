# From IPC to BNS: what it would take to keep this model useful

The Bharatiya Nyaya Sanhita, 2023 (BNS) replaced the Indian Penal Code, 1860 (IPC) on 1 July 2024. Every model in this project was trained on ILSI, whose labels are IPC sections. This section asks what survives the change, measured on our own test set.

**Source caveat.** We could not access an official Ministry of Home Affairs or BPR&D correspondence table. The mapping in `data/processed/ipc_to_bns_mapping.csv` was checked row by row against a correspondence table compiled by the Director of CAPT Bhopal (A.K. Yadav, IPS) and hosted on a Kerala government site. It is a secondary source. 96 of 100 rows are confirmed by an explicit row in that table (or, for repealed sections, by ILSI's own statute text). For the other 4 (IPC 143, 427, 294(b), 376(2)) no BNS section could be verified, so **their mapping is withheld** (`bns_sections` empty, `verified=no`) rather than guessed.

## 1. Label remapping and its limits

Of the 100 IPC labels, **84 map one-to-one** onto a single BNS section or sub-section (e.g. 302 → 103(1), 420 → 318(4), 498A → 85). Most of these changes are renumbering, sometimes with higher fines or minimum sentences. The remaining 16 do not map cleanly:

| Type | Labels | Example |
|---|---|---|
| modified (elements/scope changed) | 8 | 320 grievous hurt: the 20-day threshold became 15 days; 375: age in the marital exception 15 → 18; 380: theft now also covers means of transport and places of worship |
| one-to-many | 1 | 506 criminal intimidation splits into 351(2) and 351(3) |
| repealed long before BNS | 3 | 13 (1950), 161 and 164 (moved to the Prevention of Corruption Act, 1988) |
| unclear: mapping withheld | 4 | 143 (absent from the source table); 427 (BNS damage thresholds changed, so the sub-section depends on the amount); 294(b) and 376(2) (the table confirms only the parent sections 294 → 296 and 376 → 64) |

The clean labels carry **86.8% of all gold-label occurrences** in the test set, which sounds reassuring. But ILSI documents carry about four labels each, and a single non-clean label (most often the very frequent §506) makes the whole document ambiguous. **Only 8,272 of 13,039 test documents (63.4%) have gold labels that all map cleanly.**

Deterministic relabelling cannot change accuracy. On the clean subset, F1 is identical under IPC and BNS names (TF-IDF LR micro-F1 0.701, InLegalBERT 0.525); `src/bns_transfer.py` asserts this. Relabelling only renames the output. It does nothing about whether the model's *evidence* still holds, which is the real problem (Section 5).

## 2. Offences the model cannot predict

The BNS introduces offences with no IPC predecessor, so there are **no training examples** for them in ILSI. From the source table (`data/processed/bns_new_offences.csv`, 15 entries):
- organised crime (111), petty organised crime (112) and terrorist act (113);
- **snatching (304)**;
- murder by a group of five or more on grounds of race, caste, community and similar grounds, the "mob lynching" clause (103(2));
- the parallel group grievous-hurt clause (117(4));
- sexual intercourse by deceitful means, including a false promise of marriage (69);
- hiring a child to commit an offence (95);
- hit-and-run (106(2)), which the table notes was kept on hold rather than brought into force on 1 July 2024.

A classifier with a fixed 100-label output layer cannot name these at all. At best it will predict the nearest IPC-era neighbour (§302 for a lynching, §379 for a snatching), which is wrong in exactly the cases where the new law was meant to make a difference.

## 3. The dual-code transition period

The BNS has a repeal-and-savings clause (s. 358), and the general rule is that criminal liability is governed by the law in force when the offence was committed. In practice, offences committed before 1 July 2024 continue to be investigated, tried and appealed under the IPC. Given Indian trial and appeal timelines, IPC-labelled judgments will keep appearing for many years, alongside a growing stream of BNS cases. A deployed system therefore needs to know **which code applies to each case**, usually from the date of the offence, and should be able to output either label space. Replacing the IPC model outright would make it wrong for the backlog. Keeping it unchanged would make it wrong for new cases.

## 4. Scarcity of BNS-era training data

ILSI holds about 66,000 judgments. Very few judgments decided under the BNS will exist for some time: offences after mid-2024 must first be investigated and tried, and reported High Court and Supreme Court judgments, the kind ILSI is built from, lag trials by years. Tail labels are already a problem with 272–533 training documents each (tail-bucket F1 0.46 for TF-IDF, 0.34 for InLegalBERT). For new BNS offences the count starts at zero.

## 5. Distribution shift in how judgments are written

This is the most serious problem, and our results make it concrete. The models do not mainly infer offences from facts; they read citations.
- A regex that only extracts "section N" scores micro-F1 0.70 on IPC test data, as high as our best model.
- In the TF-IDF model, **95 of 100 labels have their own section number among their five strongest features** ("302", "section 302", "302 ipc").

BNS-era judgments will cite BNS numbers, and many of those numbers collide with unrelated IPC offences:
- **BNS 302** punishes uttering words to wound religious feelings (formerly IPC 298).
- **BNS 304** is snatching.
- **BNS 307** is theft after preparing to cause death or hurt (formerly IPC 382).

A model trained on IPC citations will read "Section 304" in a snatching judgment and predict culpable homicide with high confidence. During the transition, judgments will also cite both codes ("Section 302 IPC / 103 BNS"), mixing the signals further. On documents where no section is cited (18.9% of our test set), models are much weaker (TF-IDF 0.35, InLegalBERT 0.31 micro-F1), and that is the regime BNS-era text will push them into until they are retrained.

## 6. Three update strategies

1. **Deterministic relabelling (now, cheap).** Map predictions through the verified one-to-one table and show BNS sections alongside IPC ones (the demo app does this). Flag modified, split, merged and repealed labels instead of mapping them silently. This covers 87% of label occurrences but none of the new offences, and it is only safe if section numbers are first masked out of the input so that BNS citations cannot trigger IPC predictions.
2. **Retrain or fine-tune on BNS judgments as they accumulate.** Use a dual label space (IPC + BNS + new offences), initialise from the IPC model, and transfer each clean label's classifier weights to its BNS counterpart. Mask section numbers during training so the model learns facts → offence rather than citation → offence, which also makes it robust to the numbering change. Tail and new labels will need active collection or weak labelling from early BNS charge sheets.
3. **Retrieval / zero-shot matching against statute text.** Embed case facts and statute texts in one space and rank sections by similarity. A new offence then only needs its statutory text, not training cases. As a calibration we ran this with our fine-tuned InLegalBERT encoder (mean pooling, cosine) against the 100 **IPC** statute texts on 2,000 clean test documents: **recall@5 = 0.29**, versus 0.05 for random ranking and 0.63 for the supervised classifier on the same documents. The encoder carries real facts-to-law signal, but well below supervised performance; a retriever trained for this task (e.g. contrastive training on fact–statute pairs) should close part of the gap. The same experiment against **BNS** section texts is implemented but was not run, because no reliable machine-readable BNS text was available in this session (`data/raw/bns_sections.csv` is required).

In practice the three combine: relabel today, retrieve for offences with no data, and retrain as BNS judgments accumulate, all on number-masked text.
