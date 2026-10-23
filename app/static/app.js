// Statute Finder frontend. All DOM is built with createElement/textContent (no innerHTML of data).
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const els = {
    facts: $("facts"), words: $("wordcount"), exhibits: $("exhibits"), models: $("model-options"),
    submit: $("submit"), clear: $("clear"), error: $("error"), empty: $("empty"), results: $("results"),
    seals: $("seals"), runinfo: $("runinfo"), notices: $("notices"), ranked: $("ranked"),
  };
  const state = { meta: null, exhibit: null };
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c !== null && c !== undefined) n.append(c);
    return n;
  }
  const roman = (n) => {
    const t = [[10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
    let s = ""; for (const [v, r] of t) while (n >= v) { s += r; n -= v; } return s;
  };
  const pct = (p) => `${Math.round(p * 100)}%`;
  const countWords = (s) => (s.trim().match(/\S+/g) || []).length;
  const listSecs = (xs) => xs.map((x) => `§${x}`).join(", ");
  const shorten = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);

  function updateWords() {
    const n = countWords(els.facts.value);
    els.words.textContent = `${n.toLocaleString()} word${n === 1 ? "" : "s"}`;
  }

  // ---------------------------------------------------------------- setup
  async function init() {
    try {
      const r = await fetch("/api/meta");
      if (!r.ok) throw new Error(`server returned ${r.status}`);
      state.meta = await r.json();
    } catch (e) {
      showError(`Could not reach the model server (${e.message}). Is app/app.py running?`);
      return;
    }
    renderModels(state.meta.models);
    renderExhibits(state.meta.examples);
  }

  function renderModels(models) {
    const saved = store.get("sf-model");
    const chosen = models.some((m) => m.key === saved) ? saved : (models.find((m) => m.default) || models[0]).key;
    els.models.replaceChildren(...models.map((m) => {
      const id = `m-${m.key}`;
      const score = m.test_micro_f1 != null
        ? `test micro-F1 ${m.test_micro_f1.toFixed(3)} · macro ${m.test_macro_f1.toFixed(3)}` : "";
      return el("div", { class: "model-opt" },
        el("input", { type: "radio", name: "model", id, value: m.key, checked: m.key === chosen }),
        el("label", { for: id },
          el("span", { class: "model-name", text: m.name }),
          el("span", { class: "model-desc", text: `${score} · ${m.device.toUpperCase()}` })));
    }));
    els.models.addEventListener("change", (e) => store.set("sf-model", e.target.value));
  }

  function renderExhibits(examples) {
    els.exhibits.replaceChildren(...examples.map((ex, i) => {
      const b = el("button", { type: "button", class: "exhibit", "aria-pressed": "false",
                               title: `ILSI test document ${ex.id}` },
        el("span", { class: "ex-tag", text: `Exhibit ${"ABC"[i] || i + 1}` }),
        el("span", { text: ex.theme }));
      b.addEventListener("click", () => loadExhibit(ex, b));
      return b;
    }));
  }

  function loadExhibit(ex, btn) {
    els.facts.value = ex.text;
    state.exhibit = ex;
    for (const b of els.exhibits.querySelectorAll(".exhibit")) b.setAttribute("aria-pressed", String(b === btn));
    updateWords();
    els.facts.focus();
    els.facts.setSelectionRange(0, 0);
    els.facts.scrollTop = 0;
  }

  // ---------------------------------------------------------------- submit
  async function submit() {
    const text = els.facts.value.trim();
    hideError();
    if (countWords(text) < 5) { showError("Please enter at least a few sentences of case facts."); return; }
    const model = (document.querySelector('input[name="model"]:checked') || {}).value;
    setBusy(true);
    try {
      const r = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `server returned ${r.status}`);
      const gold = state.exhibit && state.exhibit.text.trim() === text ? state.exhibit.gold : null;
      render(data, gold);
    } catch (e) {
      showError(`Prediction failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy) {
    els.submit.disabled = busy;
    els.submit.replaceChildren(busy
      ? el("span", {}, el("span", { class: "spinner", "aria-hidden": "true" }), "Deliberating…")
      : el("span", { class: "btn-label", text: "Identify applicable sections" }));
  }
  function showError(msg) { els.error.textContent = msg; els.error.hidden = false; }
  function hideError() { els.error.hidden = true; }

  // ---------------------------------------------------------------- render
  function bnsView(row) {
    const t = row.mapping_type;
    if (t === "repealed") return [el("span", { class: "chip chip-grey", text: "Repealed · no BNS" })];
    if (!row.bns) return [el("span", { class: "chip chip-warn", text: "BNS mapping withheld" })];
    const label = row.bns.split(";").map((s) => s.trim()).join(" / ");
    const parts = [el("span", { class: "bns" }, el("small", { text: "BNS " }), label)];
    if (row.verified === "yes") parts.push(" ", el("span", { class: "chip chip-ok sm", text: "verified" }));
    if (t === "modified") parts.push(" ", el("span", { class: "chip chip-warn sm", text: "scope changed" }));
    if (t === "one_to_many") parts.push(" ", el("span", { class: "chip chip-warn sm", text: "split" }));
    return parts;
  }

  function render(data, gold) {
    els.empty.hidden = true;
    els.results.hidden = false;
    const byIpc = Object.fromEntries(data.top.map((r) => [r.ipc, r]));

    // verdict seals
    els.seals.replaceChildren(...data.predicted.map((ipc) => {
      const r = byIpc[ipc] || {};
      const bnsTxt = r.mapping_type === "repealed" ? "repealed, no BNS"
        : r.bns ? `→ BNS ${r.bns.replace(/;\s*/g, " / ")}` : "BNS mapping withheld";
      return el("div", { class: "seal" },
        el("span", { class: "seal-disc", text: `§${ipc}` }),
        el("span", { class: "seal-text" },
          el("b", { text: shorten(r.title || `IPC ${ipc}`, 46) }),
          el("span", { text: bnsTxt })));
    }));
    els.runinfo.textContent =
      `${data.model.name} · ${data.model.thresholds.replace("_", "-")} thresholds tuned on dev · ` +
      `${data.words.toLocaleString()} words · ${data.latency_ms} ms`;

    // notices
    const notes = [];
    if (data.cited_in_text.length) {
      notes.push(el("div", { class: "notice" },
        el("strong", { text: "Sections already cited in the text: " }), listSecs(data.cited_in_text) + ". ",
        "Models trained on ILSI lean heavily on such citations, so these predictions may partly echo them rather than reason from the facts."));
    }
    if (data.truncated) {
      notes.push(el("div", { class: "notice trunc" },
        el("strong", { text: "Long document: " }),
        `the model read only the first ${data.max_len} of ${data.tokens.toLocaleString()} tokens; facts or citations later in the text were not considered.`));
    }
    if (gold) {
      const hit = gold.filter((g) => data.predicted.includes(g));
      notes.push(el("div", { class: "notice gold-ref" },
        el("strong", { text: "Recorded sections for this exhibit (dataset labels): " }), listSecs(gold) + ". ",
        `The model's applied sections match ${hit.length} of ${gold.length}.`));
    }
    els.notices.replaceChildren(...notes);

    // ranked list
    els.ranked.replaceChildren(...data.top.map((r) => {
      const side = el("div", { class: "f-side" },
        r.predicted ? el("span", { class: "stamp", text: "Applies" })
                    : el("span", { class: "below", text: "below threshold" }),
        el("div", {}, ...bnsView(r)));
      const titleBits = [el("span", { class: "sec" }, "§ " + r.ipc, el("small", { text: " IPC" })),
                         el("span", { class: "ttl", text: r.title })];
      if (gold && gold.includes(r.ipc)) titleBits.push(el("span", { class: "chip chip-gold sm", text: "in record" }));
      const fill = el("span", { class: "bar-fill" });
      fill.style.width = `${(r.prob * 100).toFixed(1)}%`;
      const tick = el("span", { class: "bar-tick", title: `threshold ${r.threshold}` });
      tick.style.left = `calc(${Math.min(100, Math.max(0, r.threshold * 100))}% - 1px)`;
      const item = el("li", { class: `finding${r.predicted ? " applies" : ""}` },
        el("span", { class: "numeral", text: roman(r.rank), "aria-label": `Rank ${r.rank}` }),
        el("div", { class: "f-main" }, el("p", { class: "f-title" }, ...titleBits)),
        side,
        el("div", { class: "bar-row" },
          el("div", { class: "bar", role: "img", "aria-label": `score ${pct(r.prob)}, threshold ${pct(r.threshold)}` },
             fill, tick),
          el("span", { class: "pct", text: pct(r.prob) })));
      if (r.notes && (r.mapping_type !== "one_to_one" || r.verified !== "yes")) {
        item.append(el("details", { class: "note" }, el("summary", { text: "About this mapping" }),
                       el("p", { text: r.notes })));
      }
      return item;
    }));
    if (window.matchMedia("(max-width: 920px)").matches) els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------------------------------------------------------------- events
  els.facts.addEventListener("input", () => {
    updateWords();
    if (state.exhibit && els.facts.value.trim() !== state.exhibit.text.trim()) {
      for (const b of els.exhibits.querySelectorAll(".exhibit")) b.setAttribute("aria-pressed", "false");
    }
  });
  els.facts.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submit(); }
  });
  els.submit.addEventListener("click", submit);
  els.clear.addEventListener("click", () => {
    els.facts.value = ""; state.exhibit = null; updateWords(); hideError();
    for (const b of els.exhibits.querySelectorAll(".exhibit")) b.setAttribute("aria-pressed", "false");
    els.results.hidden = true; els.empty.hidden = false; els.facts.focus();
  });

  // theme: dark by default, remembered per browser
  const themeBtn = $("theme-toggle");
  const paintTheme = () => {
    const dark = document.documentElement.dataset.theme !== "light";
    themeBtn.textContent = dark ? "☀ Light" : "☾ Dark";
    themeBtn.setAttribute("aria-pressed", String(!dark));
  };
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    store.set("sf-theme", next);
    paintTheme();
  });
  paintTheme();

  updateWords();
  init();
})();
