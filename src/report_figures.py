"""Build every figure and table used by report/REPORT.md into report/figures and report/tables.

Reads only saved artefacts (no training). Colours: one fixed colour per model across all
figures (validated categorical palette, slot order); identity is never colour-alone
(legends + direct labels, and every chart has a matching table in report/tables).

Usage: python src/report_figures.py
"""
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as A  # noqa: E402
from common import PROC, ROOT  # noqa: E402

REPORT = ROOT / "report"
FIG, TAB = REPORT / "figures", REPORT / "tables"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"

# one colour per model, everywhere (palette slots 1-6, light mode, validated adjacent-pairs)
MODELS = {
    "tfidf_lr": ("TF-IDF + LR", "#2a78d6"),
    "inlegalbert_head_512": ("InLegalBERT (head)", "#eb6834"),
    "regex_mention": ("Citation diagnostic", "#1baf7a"),
    "bert-base-uncased_head_512": ("BERT-base (head)", "#eda100"),
    "inlegalbert_head_tail_512": ("InLegalBERT (head+tail)", "#e87ba4"),
    "keyword": ("Keyword", "#008300"),
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "text.color": INK,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure", name)


def table(df, name, floatfmt="{:.3f}"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    md = "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
                   + ["| " + " | ".join(r) + " |" for r in cells])
    (TAB / f"{name}.md").write_text(md + "\n")
    print("table ", name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    ctx = A.load_context()
    res = ctx["results"]
    sel = A.dev_selected_rows(res)
    order = [m for m in MODELS if m in set(sel.model)]
    sel = sel.set_index("model").loc[order].reset_index()

    # ---- EDA figures (copied from outputs/eda)
    for f in ["label_frequency.png", "labels_per_doc.png", "doc_lengths.png", "cooccurrence_top20.png", "label_buckets.png"]:
        shutil.copy2(ROOT / "outputs" / "eda" / f, FIG / f"eda_{f}")
        print("figure eda_" + f)

    # ---- results tables
    t = sel[["model", "thresholds", "test_micro_f1", "test_macro_f1", "test_micro_p", "test_micro_r",
             "test_samples_f1", "test_head_f1", "test_mid_f1", "test_tail_f1", "dev_micro_f1"]].copy()
    t.insert(0, "Model", [MODELS[m][0] for m in t.model])
    t = t.drop(columns="model")
    t.columns = ["Model", "Thresholds", "Micro-F1", "Macro-F1", "Micro-P", "Micro-R", "Samples-F1",
                 "Head F1", "Mid F1", "Tail F1", "Dev micro-F1"]
    table(t, "results_main")
    full = res[res.model.isin(MODELS)].copy()
    full.insert(0, "Model", [MODELS[m][0] for m in full.model])
    full = full[["Model", "thresholds", "test_micro_f1", "test_macro_f1", "test_micro_p", "test_micro_r",
                 "test_samples_f1", "dev_micro_f1", "dev_macro_f1"]]
    full.columns = ["Model", "Thresholds", "Micro-F1", "Macro-F1", "Micro-P", "Micro-R", "Samples-F1",
                    "Dev micro-F1", "Dev macro-F1"]
    table(full, "results_all_strategies")

    # ---- F1: results overview (dot plot; filled = micro, hollow = macro; colour = model)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(sel))[::-1]
    for yi, (_, r) in zip(y, sel.iterrows()):
        col = MODELS[r.model][1]
        ax.plot([r.test_macro_f1, r.test_micro_f1], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(r.test_micro_f1, yi, s=70, color=col, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter(r.test_macro_f1, yi, s=70, facecolor="white", edgecolor=col, linewidth=2, zorder=3)
        ax.text(r.test_micro_f1 + 0.012, yi, f"{r.test_micro_f1:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.text(r.test_macro_f1 - 0.012, yi, f"{r.test_macro_f1:.3f}", va="center", ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([MODELS[m][0] + (" *" if m == "regex_mention" else "") for m in sel.model])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Test F1 (each model at its dev-selected threshold strategy)")
    ax.scatter([], [], s=60, color=INK2, label="micro-F1")
    ax.scatter([], [], s=60, facecolor="white", edgecolor=INK2, linewidth=2, label="macro-F1")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Test micro- and macro-F1 by model    (* diagnostic, not a model)", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_results_overview.png")

    # ---- F2: threshold strategy effect
    strat = ["t0.5", "global", "per_label"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for m in ["tfidf_lr", "inlegalbert_head_512"]:
        name, col = MODELS[m]
        v = [float(res[(res.model == m) & (res.thresholds == s)].test_micro_f1.iloc[0]) for s in strat]
        ax.plot(range(3), v, color=col, lw=2, marker="o", ms=8, markeredgecolor="white", markeredgewidth=1.5, label=name)
        for i, val in enumerate(v):
            ax.text(i, val + 0.018, f"{val:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["fixed 0.5", "one global\n(tuned on dev)", "per-label\n(tuned on dev)"])
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("Test micro-F1")
    ax.legend(loc="lower right")
    ax.set_title("Effect of the threshold strategy", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_thresholds.png")

    # ---- F3: bucket F1
    bk = sel.set_index("model")[["test_head_f1", "test_mid_f1", "test_tail_f1"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    n = len(bk)
    w = 0.8 / n
    for i, m in enumerate(bk.index):
        name, col = MODELS[m]
        ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, bk.loc[m].values, w * 0.92, color=col, label=name,
               edgecolor="white", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Head (34 labels)", "Mid (33)", "Tail (33)"])
    ax.set_ylabel("Mean per-label test F1")
    ax.set_ylim(0, 0.85)
    ax.legend(ncol=3, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_title("F1 by label-frequency bucket (train-frequency tertiles)", loc="left")
    save(fig, "fig_buckets.png")
    tb = bk.reset_index()
    tb.insert(0, "Model", [MODELS[m][0] for m in tb.model])
    tb = tb.drop(columns="model")
    tb.columns = ["Model", "Head F1", "Mid F1", "Tail F1"]
    table(tb, "bucket_f1")

    # ---- F4: per-label F1 vs training frequency
    base, bert = A.load_model(ctx, "best_baseline"), A.load_model(ctx, "best_bert")
    freq = ctx["train_counts"].to_numpy()
    f1 = {m.name: A.per_label_f1(ctx["y_test"], m.pred) for m in [base, bert]}
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for m in [base, bert]:
        name, col = MODELS[m.name]
        r = np.corrcoef(np.log10(freq), f1[m.name])[0, 1]
        ax.scatter(freq, f1[m.name], s=26, color=col, alpha=0.85, edgecolor="white", linewidth=0.6,
                   label=f"{name} (r = {r:.2f})")
    offsets = {"2": (-16, -4), "4": (5, -11)}
    for lab in ["2", "3", "4", "34", "302", "450"]:
        j = ctx["label_list"].index(lab)
        ax.annotate(f"§{lab}", (freq[j], f1[bert.name][j]), xytext=offsets.get(lab, (4, -10)),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Training documents with the label (log scale)")
    ax.set_ylabel("Per-label test F1")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Per-label F1 vs label frequency (r: Pearson with log frequency)", loc="left")
    save(fig, "fig_f1_vs_frequency.png")
    pl = pd.DataFrame({"IPC section": ctx["label_list"], "Train docs": freq,
                       "F1 TF-IDF+LR": f1[base.name], "F1 InLegalBERT": f1[bert.name]})
    table(pl.sort_values("Train docs", ascending=False), "per_label_f1")

    # ---- F5: F1 by citation position
    _, pos = A.mention_position_stats(ctx)
    regex = A.Model("regex_mention", "outputs/baselines/regex_mention", "global", ctx["label_list"])
    cp = A.f1_by_mention_group(ctx, [base, bert, regex], pos)
    ms = ["tfidf_lr", "inlegalbert_head_512", "regex_mention"]
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, m in enumerate(ms):
        name, col = MODELS[m]
        bars = ax.bar(np.arange(3) + (i - 1) * w, cp[m], w * 0.92, color=col, label=name, edgecolor="white", linewidth=1)
        for b, v in zip(bars, cp[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"cited in first ~400 words\n({cp.docs[0]:,} docs)", f"cited only later\n({cp.docs[1]:,} docs)",
                        f"never cited\n({cp.docs[2]:,} docs)"])
    ax.set_ylabel("Test micro-F1")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_title("Where is a gold section first cited in the judgment?", loc="left")
    save(fig, "fig_citation_position.png")
    table(cp.rename(columns={"group": "Documents where a gold section is…", "docs": "Docs",
                             **{m: MODELS[m][0] for m in ms}}), "citation_position")

    # ---- F6: BERT training curves (small multiples; overlaid lines failed the colour check)
    runs = ["inlegalbert_head_512", "bert-base-uncased_head_512", "inlegalbert_head_tail_512"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 4.6), sharex=True, sharey="row")
    rows = []
    for c, run in enumerate(runs):
        name, col = MODELS[run]
        log = pd.read_csv(ROOT / "colab_downloads" / run / "train_log.csv")
        axes[0, c].plot(log.epoch, log.train_loss, color=col, lw=2, marker="o", ms=5)
        axes[1, c].plot(log.epoch, log.eval_micro_f1_t05, color=col, lw=2, marker="o", ms=5)
        b = int(log.eval_micro_f1_t05.idxmax())
        axes[1, c].scatter(log.epoch[b], log.eval_micro_f1_t05[b], s=90, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=4)
        axes[1, c].annotate("kept", (log.epoch[b], log.eval_micro_f1_t05[b]), xytext=(0, -16),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        axes[0, c].set_title(name, fontsize=10, color=INK)
        axes[1, c].set_xlabel("epoch")
        axes[1, c].set_xticks([1, 2, 3, 4])
        for _, r in log.iterrows():
            rows.append({"Run": name, "Epoch": int(r.epoch), "Train loss": r.train_loss, "Dev loss": r.eval_loss,
                         "Dev micro-F1 @0.5": r.eval_micro_f1_t05, "Dev macro-F1 @0.5": r.eval_macro_f1_t05})
    axes[0, 0].set_ylabel("training loss")
    axes[1, 0].set_ylabel("dev micro-F1 @ 0.5")
    fig.suptitle("BERT fine-tuning curves (A100, ~13 min per run)", x=0.02, ha="left", fontweight="bold", fontsize=11.5)
    save(fig, "fig_training_curves.png")
    table(pd.DataFrame(rows), "training_log")

    # ---- F7: confusion heat-maps for the focus families
    focus = ["34", "147", "148", "149", "302", "304", "307", "323", "324", "325", "406", "420"]
    idx = [ctx["label_list"].index(s) for s in focus]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, m in zip(axes, [base, bert]):
        yv = ctx["y_test"].astype(np.int32)
        pv = m.pred.astype(np.int32)
        C = ((pv * (1 - yv)).T @ yv)[np.ix_(idx, idx)] / np.maximum(yv.sum(0)[idx], 1)
        C = C.astype(float)
        np.fill_diagonal(C, np.nan)
        im = ax.imshow(C, cmap="Blues", vmin=0, vmax=0.5)
        for i in range(len(idx)):
            for j in range(len(idx)):
                if i != j and C[i, j] >= 0.08:
                    ax.text(j, i, f"{C[i, j]:.2f}"[1:], ha="center", va="center", fontsize=6.5,
                            color="white" if C[i, j] > 0.25 else INK)
        ax.set_xticks(range(len(idx)))
        ax.set_yticks(range(len(idx)))
        ax.set_xticklabels(focus, fontsize=8)
        ax.set_yticklabels(focus, fontsize=8)
        ax.set_xlabel("gold section (true)")
        ax.set_ylabel("falsely predicted section")
        ax.set_title(MODELS[m.name][0], fontsize=10.5)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("share of documents with the gold section")
    fig.suptitle("False positives among related sections (rows) given a true section (columns)", x=0.02, ha="left",
                 fontweight="bold", fontsize=11.5)
    fig.savefig(FIG / "fig_confusions.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure fig_confusions.png")
    shutil.copy2(ROOT / "outputs" / "analysis" / "focus_confusions.csv", TAB / "focus_confusions.csv")

    # ---- F8: BNS mapping coverage
    cov = json.loads((ROOT / "outputs" / "bns" / "remap_metrics.json").read_text())["coverage"]
    mp = pd.read_csv(PROC / "ipc_to_bns_mapping.csv", dtype=str).fillna("")
    types = ["one_to_one", "modified", "one_to_many", "repealed", "unclear"]
    tcolors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    tnames = {"one_to_one": "one-to-one", "modified": "scope changed", "one_to_many": "split",
              "repealed": "repealed", "unclear": "withheld (unverifiable)"}
    counts = mp.mapping_type.value_counts().reindex(types, fill_value=0) / len(mp)
    mass = pd.Series(cov["test_label_mass_by_type"]).reindex(types, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 2.7))
    for yi, ser in enumerate([mass, counts]):
        left = 0.0
        for t_, col in zip(types, tcolors):
            v = float(ser[t_])
            ax.barh(yi, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.62,
                    label=tnames[t_] if yi == 0 else None)
            if v >= 0.04:
                ax.text(left + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if t_ in ("one_to_one", "modified") else INK)
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["share of test\nlabel occurrences", "share of the\n100 labels"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title(f"IPC → BNS mapping of the label set ({cov['test_docs_all_gold_clean']:,} of "
                 f"{cov['test_docs_total']:,} test documents map fully cleanly)", loc="left", fontsize=10.5)
    save(fig, "fig_bns_coverage.png")
    shutil.copy2(PROC / "ipc_to_bns_mapping.csv", TAB / "ipc_to_bns_mapping.csv")
    shutil.copy2(PROC / "bns_new_offences.csv", TAB / "bns_new_offences.csv")
    table(pd.DataFrame([{"Mapping type": tnames[t_], "Labels": int(round(counts[t_] * len(mp))),
                         "Share of test label occurrences": float(mass[t_])} for t_ in types]), "bns_coverage")

    # ---- F9: zero-shot retrieval
    zs = json.loads((ROOT / "outputs" / "bns" / "zero_shot.json").read_text())["ipc_statute_retrieval"]
    vals = [("random ranking", zs["random_recall@5"], "#c3c2b7", None),
            ("zero-shot: facts ↔ statute text\n(fine-tuned InLegalBERT encoder)", zs["recall@5"], "#eb6834", "//"),
            ("supervised classifier\n(InLegalBERT)", zs["supervised_classifier_recall@5 (same docs)"], "#eb6834", None)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (lab, v, col, hatch) in enumerate(vals):
        ax.barh(i, v, color=col, height=0.6, hatch=hatch, edgecolor="white")
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(3))
    ax.set_yticklabels([v[0] for v in vals], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("recall@5 over the 100 IPC sections (2,000 test documents)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Retrieving statutes by text similarity (calibration for the BNS experiment)", loc="left", fontsize=10.5)
    save(fig, "fig_zero_shot.png")

    # ---- remaining tables
    bs = json.loads((ROOT / "outputs" / "analysis" / "bootstrap.json").read_text())
    table(pd.DataFrame([{"Metric": k.replace("_", "-"), "TF-IDF + LR [95% CI]": f"{v['a']:.3f} {v['a_CI95']}",
                         "InLegalBERT [95% CI]": f"{v['b']:.3f} {v['b_CI95']}",
                         "Difference [95% CI]": f"{v['diff_b_minus_a']:+.3f} {v['diff_CI95']}",
                         "p (two-sided)": str(v["p_two_sided"])}
                        for k, v in bs.items() if k in ("micro_f1", "macro_f1")]), "bootstrap")
    for f in ["bert_largest_gains.csv", "bert_largest_losses.csv", "errors_bert_failed.csv",
              "errors_baseline_failed.csv", "confusions_tfidf_lr.csv", "confusions_inlegalbert_head_512.csv"]:
        shutil.copy2(ROOT / "outputs" / "analysis" / f, TAB / f)
    buckets = json.loads((PROC / "label_buckets.json").read_text())
    lab2b = {l: b for b in ("head", "mid", "tail") for l in buckets[b]}
    labs = mp[["ipc_section", "ipc_title", "bns_sections", "mapping_type", "verified"]].copy()
    labs.insert(2, "train_docs", [buckets["train_counts"][s] for s in labs.ipc_section])
    labs.insert(3, "bucket", [lab2b[s] for s in labs.ipc_section])
    table(labs, "label_set")
    print(f"done -> {FIG} and {TAB}")


if __name__ == "__main__":
    main()
