/* Signature Lab: every number on the page comes from data/site.json,
   which scripts/export_site_data.py builds from docs/results.json. */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const KEYS = [
    { ch: ".", name: "period" }, { ch: "t", name: "t" }, { ch: "i", name: "i" },
    { ch: "e", name: "e" }, { ch: "5", name: "five" }, { ch: "R", name: "Shift.r" },
    { ch: "o", name: "o" }, { ch: "a", name: "a" }, { ch: "n", name: "n" },
    { ch: "l", name: "l" }, { ch: "Enter", name: "Return", label: "⏎" },
  ];
  const DETECTOR_LABELS = {
    scaled_manhattan: "Scaled Manhattan",
    isolation_forest: "Isolation Forest",
    one_class_svm: "One-class SVM",
    autoencoder: "Autoencoder",
    fusion: "Fusion of 4",
  };

  /* ---------- small helpers ---------- */
  const $ = (sel) => document.querySelector(sel);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d) : "-");
  const pct = (v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "-");
  function get(obj, path) {
    return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
  }
  function el(tag, attrs, parent, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
    return null;
  }

  /* ---------- theme ---------- */
  const saved = store("kma-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("#theme").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    store("kma-theme", next);
    if (DATA) drawAll();
  });

  let DATA = null;
  const state = { features: null, score: null, attempts: 0, channel: "mouse", driftUser: null };

  fetch("data/site.json")
    .then((r) => r.json())
    .then((data) => {
      DATA = data;
      fillValues();
      setupDemo();
      setupPad();
      setupDrift();
      fillTables();
      drawAll();
      let resizeTimer;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(drawPad, 120);
      });
    })
    .catch(() => {
      $("#status").textContent = "Could not load data/site.json.";
      $("#status").classList.add("err");
    });

  function fillValues() {
    document.querySelectorAll("[data-v]").forEach((node) => {
      const value = get(DATA, node.dataset.v);
      node.textContent = fmt(value, Number(node.dataset.f || 3));
    });
  }

  function drawAll() {
    drawEerChart();
    drawActionsChart();
    drawMouseChart();
    drawRiskCards();
    drawDrift();
    drawAging();
    drawHist();
    drawHolds();
  }

  /* =========================================================
     Typing demo
     ========================================================= */
  const typing = { downs: [], ups: [], index: 0, active: false };

  function setupDemo() {
    const prompt = $("#prompt");
    KEYS.forEach((k, i) => {
      const span = document.createElement("span");
      span.className = "k" + (i === 0 ? " next" : "");
      span.innerHTML = (k.label || k.ch) + "<small>&nbsp;</small>";
      prompt.appendChild(span);
    });
    const labels = $("#holdLabels");
    KEYS.forEach((k) => {
      const d = document.createElement("div");
      d.textContent = k.label || k.ch;
      labels.appendChild(d);
    });
    const select = $("#claim");
    Object.keys(DATA.keystroke.templates).sort().forEach((user) => {
      const opt = document.createElement("option");
      opt.value = user;
      opt.textContent = user;
      select.appendChild(opt);
    });
    select.value = store("kma-claim") || "s002";
    select.addEventListener("change", () => {
      store("kma-claim", select.value);
      scoreAttempt();
      drawHolds();
      drawHist();
    });

    const input = $("#typeInput");
    input.addEventListener("keydown", onKeyDown);
    input.addEventListener("keyup", onKeyUp);
    input.addEventListener("blur", () => { if (typing.active) resetTyping("Focus left the box. Start again."); });
  }

  function resetTyping(message, isError) {
    typing.downs = [];
    typing.ups = [];
    typing.index = 0;
    typing.active = false;
    $("#typeInput").value = "";
    document.querySelectorAll("#prompt .k").forEach((node, i) => {
      node.className = "k" + (i === 0 ? " next" : "");
      node.querySelector("small").innerHTML = "&nbsp;";
    });
    const status = $("#status");
    status.textContent = message || "Timings never leave your browser.";
    status.classList.toggle("err", Boolean(isError));
  }

  function matches(key, expected) {
    if (expected === "R") return key === "R" || key === "r";
    return key === expected;
  }

  function onKeyDown(e) {
    if (e.key === "Unidentified" || e.key === "Process" || e.isComposing) {
      resetTyping("This demo needs a physical keyboard; on-screen keyboards do not report key timing.", true);
      return;
    }
    if (e.repeat || e.key === "Shift" || e.key === "CapsLock") return;
    if (e.key === "Enter") e.preventDefault();
    if (e.key === "Tab") return;
    if (typing.index >= KEYS.length) return;
    const expected = KEYS[typing.index].ch;
    if (!matches(e.key, expected)) {
      e.preventDefault();
      resetTyping(e.key === "Backspace" ? "Cleared. Start again from the period." : `Expected "${expected === "Enter" ? "Enter" : expected}". Start again from the period.`, e.key !== "Backspace");
      return;
    }
    if (typing.index === 0) {
      typing.active = true;
      $("#status").textContent = "Recording…";
      $("#status").classList.remove("err");
    }
    typing.downs[typing.index] = e.timeStamp;
    const nodes = document.querySelectorAll("#prompt .k");
    nodes[typing.index].className = "k down";
    if (nodes[typing.index + 1]) nodes[typing.index + 1].classList.add("next");
    typing.index += 1;
  }

  function onKeyUp(e) {
    if (e.key === "Shift" || e.key === "CapsLock") return;
    const key = e.key === "r" ? "R" : e.key;
    for (let i = 0; i < typing.index; i++) {
      if (typing.ups[i] == null && matches(key, KEYS[i].ch)) {
        typing.ups[i] = e.timeStamp;
        const node = document.querySelectorAll("#prompt .k")[i];
        node.className = "k done";
        node.querySelector("small").textContent = Math.round(typing.ups[i] - typing.downs[i]);
        break;
      }
    }
    if (typing.index === KEYS.length && typing.ups.filter((v) => v != null).length === KEYS.length) {
      finishAttempt();
    }
  }

  function finishAttempt() {
    const at = (name) => KEYS.findIndex((k) => k.name === name);
    const features = DATA.keystroke.feature_names.map((name) => {
      const parts = name.split(".");
      const kind = parts[0];
      if (kind === "H") {
        const i = at(parts.slice(1).join("."));
        return (typing.ups[i] - typing.downs[i]) / 1000;
      }
      // DD.a.b and UD.a.b, where "Shift.r" is one key name
      const rest = parts.slice(1).join(".");
      let a, b;
      for (let i = 0; i < KEYS.length - 1; i++) {
        if (rest === `${KEYS[i].name}.${KEYS[i + 1].name}`) { a = i; b = i + 1; }
      }
      if (kind === "DD") return (typing.downs[b] - typing.downs[a]) / 1000;
      return (typing.downs[b] - typing.ups[a]) / 1000;
    });
    if (features.some((v) => !Number.isFinite(v))) {
      resetTyping("Could not read the timing. Try again.", true);
      return;
    }
    state.features = features;
    state.attempts += 1;
    $("#attempts").textContent = `${state.attempts} attempt${state.attempts === 1 ? "" : "s"}`;
    typing.active = false;
    $("#typeInput").value = "";
    typing.downs = [];
    typing.ups = [];
    typing.index = 0;
    document.querySelectorAll("#prompt .k").forEach((node, i) => { node.classList.toggle("next", i === 0); });
    $("#status").textContent = "Scored. Type it again to see how much you vary.";
    $("#status").classList.remove("err");
    scoreAttempt();
    drawHolds();
    drawHist();
  }

  function scaledManhattan(features, template) {
    let total = 0;
    for (let i = 0; i < features.length; i++) {
      total += Math.abs(features[i] - template.center[i]) / template.scale[i];
    }
    return total / features.length;
  }

  function scoreAttempt() {
    if (!state.features) return;
    const user = $("#claim").value;
    const template = DATA.keystroke.templates[user];
    const score = scaledManhattan(state.features, template);
    state.score = score;
    const accept = score <= template.threshold;
    const badge = $("#badge");
    badge.className = "badge " + (accept ? "accept" : "reject");
    badge.textContent = accept ? "accepted" : "rejected";
    const ratio = score / template.threshold;
    $("#verdictText").innerHTML = accept
      ? `Score <b class="mono">${fmt(score, 2)}</b> is under ${user}'s threshold of <b class="mono">${fmt(template.threshold, 2)}</b>. You typed close enough to pass as ${user}.`
      : `Score <b class="mono">${fmt(score, 2)}</b> against a threshold of <b class="mono">${fmt(template.threshold, 2)}</b>: ${fmt(ratio, 1)}x further from ${user}'s template than ${user}'s own usual attempts.`;
    const ranked = Object.entries(DATA.keystroke.templates)
      .map(([id, t]) => [id, scaledManhattan(state.features, t)])
      .sort((a, b) => a[1] - b[1]);
    const [bestId, bestScore] = ranked[0];
    const passes = ranked.filter(([id, s]) => s <= DATA.keystroke.templates[id].threshold).length;
    $("#closest").textContent = `${bestId} (score ${fmt(bestScore, 2)}). You would pass as ${passes} of 51 users.`;
  }

  function drawHolds() {
    const box = $("#holds");
    box.innerHTML = "";
    const template = DATA.keystroke.templates[$("#claim").value];
    const names = DATA.keystroke.feature_names;
    const refHolds = KEYS.map((k) => template.center[names.indexOf("H." + k.name)]);
    const mine = state.features ? KEYS.map((k) => state.features[names.indexOf("H." + k.name)]) : null;
    const top = Math.max(...refHolds, ...(mine || [0])) * 1.1;
    KEYS.forEach((k, i) => {
      const col = document.createElement("div");
      col.style.background = "transparent";
      col.style.height = "100%";
      col.style.opacity = "1";
      const ref = document.createElement("div");
      ref.className = "ref";
      ref.style.height = (refHolds[i] / top) * 100 + "%";
      col.appendChild(ref);
      if (mine) {
        const bar = document.createElement("div");
        bar.style.position = "absolute";
        bar.style.left = "18%";
        bar.style.right = "18%";
        bar.style.bottom = "0";
        bar.style.height = (Math.max(mine[i], 0) / top) * 100 + "%";
        bar.title = `${k.label || k.ch}: you ${Math.round(mine[i] * 1000)} ms, average ${Math.round(refHolds[i] * 1000)} ms`;
        col.appendChild(bar);
      }
      box.appendChild(col);
    });
  }

  function drawHist() {
    const svg = $("#demoHist");
    clear(svg);
    const h = DATA.keystroke.scaled_manhattan_histogram;
    const W = 560, H = 150, L = 8, R = 8, T = 10, B = 24;
    const n = h.genuine.length;
    const gTotal = h.genuine.reduce((a, b) => a + b, 0);
    const iTotal = h.impostor.reduce((a, b) => a + b, 0);
    const g = h.genuine.map((v) => v / gTotal);
    const im = h.impostor.map((v) => v / iTotal);
    const top = Math.max(...g, ...im);
    const xMax = h.edges[n];
    const x = (v) => L + (Math.min(v, xMax) / xMax) * (W - L - R);
    const y = (v) => H - B - (v / top) * (H - T - B);
    const bw = (W - L - R) / n;
    for (let i = 0; i < n; i++) {
      el("rect", { x: L + i * bw, y: y(g[i]), width: bw - 1, height: H - B - y(g[i]), fill: css("--genuine"), opacity: 0.55 }, svg);
      el("rect", { x: L + i * bw, y: y(im[i]), width: bw - 1, height: H - B - y(im[i]), fill: css("--impostor"), opacity: 0.45 }, svg);
    }
    el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
    for (let t = 0; t <= xMax; t += 1) {
      el("text", { x: x(t), y: H - 8, "text-anchor": "middle" }, svg, String(t));
    }
    el("text", { x: W - R, y: T + 4, "text-anchor": "end" }, svg, "scaled Manhattan score, all 51 users");
    if (state.score != null) {
      const sx = x(state.score);
      el("line", { x1: sx, x2: sx, y1: T, y2: H - B, stroke: css("--accent"), "stroke-width": 2 }, svg);
      el("text", { x: Math.min(sx + 4, W - 60), y: T + 18, fill: css("--accent") }, svg, state.score > xMax ? `you: ${fmt(state.score, 1)} →` : "you");
    }
  }

  /* =========================================================
     Charts
     ========================================================= */
  function axisY(svg, y, ticks, L, W, R, format) {
    ticks.forEach((t) => {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: "gridline" }, svg);
      el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, svg, format ? format(t) : String(t));
    });
  }

  function drawEerChart() {
    const svg = $("#eerChart");
    clear(svg);
    const W = 560, H = 300, L = 44, R = 10, T = 14, B = 44;
    const names = Object.keys(DETECTOR_LABELS).filter((n) => DATA.keystroke.per_user_eer[n]);
    const all = names.flatMap((n) => DATA.keystroke.per_user_eer[n]);
    const yMax = Math.ceil(Math.max(...all) * 10) / 10;
    const y = (v) => H - B - (v / yMax) * (H - T - B);
    const ticks = [];
    for (let t = 0; t <= yMax + 1e-9; t += 0.1) ticks.push(Math.round(t * 10) / 10);
    axisY(svg, y, ticks, L, W, R, (t) => t.toFixed(1));
    const band = (W - L - R) / names.length;
    const bestName = names.filter((n) => n !== "fusion").reduce((a, b) =>
      DATA.keystroke.summary[a].mean_eer < DATA.keystroke.summary[b].mean_eer ? a : b);
    names.forEach((name, i) => {
      const cx = L + band * (i + 0.5);
      const color = name === bestName ? css("--accent") : css("--genuine");
      DATA.keystroke.per_user_eer[name].forEach((v, j) => {
        const jitter = (((j * 37) % 51) / 51 - 0.5) * band * 0.55;
        el("circle", { cx: cx + jitter, cy: y(v), r: 2.6, fill: color, opacity: 0.45 }, svg);
      });
      const mean = DATA.keystroke.summary[name].mean_eer;
      el("line", { x1: cx - band * 0.34, x2: cx + band * 0.34, y1: y(mean), y2: y(mean), stroke: css("--ink"), "stroke-width": 2.5 }, svg);
      el("text", { x: cx, y: y(mean) - 7, "text-anchor": "middle", fill: css("--ink") }, svg, mean.toFixed(3));
      const label = DETECTOR_LABELS[name].split(" ");
      el("text", { x: cx, y: H - B + 18, "text-anchor": "middle" }, svg, label[0]);
      if (label[1]) el("text", { x: cx, y: H - B + 32, "text-anchor": "middle" }, svg, label.slice(1).join(" "));
    });
    const paper = DATA.keystroke.paper_scaled_manhattan_eer;
    el("line", { x1: L, x2: W - R, y1: y(paper), y2: y(paper), stroke: css("--accent"), "stroke-dasharray": "5 4", "stroke-width": 1.2 }, svg);
    el("text", { x: W - R, y: y(paper) - 5, "text-anchor": "end", fill: css("--accent") }, svg, "paper 0.096");
  }

  function drawActionsChart() {
    const svg = $("#actionsChart");
    clear(svg);
    const W = 560, H = 260, L = 44, R = 16, T = 16, B = 40;
    const rows = DATA.mouse.eer_vs_actions;
    const yMax = 0.4;
    const y = (v) => H - B - (v / yMax) * (H - T - B);
    const x = (i) => L + (i / (rows.length - 1)) * (W - L - R);
    axisY(svg, y, [0, 0.1, 0.2, 0.3, 0.4], L, W, R, (t) => t.toFixed(1));
    const d = rows.map((r, i) => `${i ? "L" : "M"}${x(i)},${y(r.eer)}`).join(" ");
    el("path", { d, fill: "none", stroke: css("--accent"), "stroke-width": 2 }, svg);
    rows.forEach((r, i) => {
      el("circle", { cx: x(i), cy: y(r.eer), r: 4, fill: css("--accent") }, svg);
      el("text", { x: x(i), y: y(r.eer) - 10, "text-anchor": "middle", fill: css("--ink") }, svg, r.eer.toFixed(3));
      el("text", { x: x(i), y: H - B + 18, "text-anchor": "middle" }, svg, String(r.actions));
    });
    el("text", { x: (W + L) / 2, y: H - 6, "text-anchor": "middle" }, svg, "actions averaged per session");
    el("text", { x: L, y: T - 2 }, svg, "EER");
  }

  function drawMouseChart() {
    const svg = $("#mouseChart");
    clear(svg);
    const W = 1100, H = 230, L = 44, R = 10, T = 18, B = 46;
    const rows = DATA.mouse_per_user;
    const lo = 0.8;
    const y = (v) => H - B - ((v - lo) / (1 - lo)) * (H - T - B);
    axisY(svg, y, [0.8, 0.85, 0.9, 0.95, 1.0], L, W, R, (t) => t.toFixed(2));
    const band = (W - L - R) / rows.length;
    rows.forEach((r, i) => {
      const x0 = L + band * i + band * 0.22;
      const w = band * 0.56;
      el("rect", { x: x0, y: y(r.auc), width: w, height: H - B - y(r.auc), fill: css("--genuine"), opacity: 0.8, rx: 2 }, svg);
      el("text", { x: x0 + w / 2, y: y(r.auc) - 6, "text-anchor": "middle", fill: css("--ink") }, svg, r.auc.toFixed(3));
      el("text", { x: x0 + w / 2, y: H - B + 16, "text-anchor": "middle" }, svg, r.user_id);
      el("text", { x: x0 + w / 2, y: H - B + 31, "text-anchor": "middle" }, svg, `EER ${r.eer.toFixed(2)}`);
    });
    el("text", { x: L, y: T - 6 }, svg, "AUC");
  }

  function drawRiskCards() {
    const box = $("#riskCards");
    box.innerHTML = "";
    DATA.risk_examples.forEach((ex) => {
      const truth = ex.label ? "impostor" : "genuine";
      const color = ex.risk >= 50 ? "var(--impostor)" : "var(--genuine)";
      const card = document.createElement("div");
      card.className = "card risk-card";
      const flags = Object.entries(ex.flags)
        .map(([k, v]) => `<span class="flag${v ? " on" : ""}">${k.replace("_", " ")}${v ? " on" : " off"}</span>`)
        .join("");
      card.innerHTML = `
        <div class="meta">${ex.title}</div>
        <div class="top"><span class="score" style="color:${color}">${ex.risk.toFixed(1)}</span>
          <span class="meta">claimed ${ex.user_id} · truly <span class="${ex.label ? "i" : "g"}">${truth}</span></span></div>
        <div class="meter"><i style="width:${Math.max(ex.risk, 1)}%;background:${color}"></i></div>
        <ul class="reasons">${ex.reasons.map((r) => `<li>${r}</li>`).join("")}</ul>
        <div class="flags">${flags}<span class="flag">synthetic</span></div>
        <p class="note" style="margin-top:8px">${ex.actions} actions · anomaly ${ex.behavioral_anomaly.toFixed(3)}</p>`;
      box.appendChild(card);
    });
  }

  function fillTables() {
    const body = $("#keyTable tbody");
    const names = Object.keys(DETECTOR_LABELS).filter((n) => DATA.keystroke.summary[n]);
    const best = names.reduce((a, b) => (DATA.keystroke.summary[a].mean_eer < DATA.keystroke.summary[b].mean_eer ? a : b));
    names.forEach((n) => {
      const s = DATA.keystroke.summary[n];
      const tr = document.createElement("tr");
      if (n === best) tr.className = "best";
      tr.innerHTML = `<td>${DETECTOR_LABELS[n]}</td><td class="num">${fmt(s.mean_eer, 4)}</td>
        <td class="num">${pct(s.mean_far_at_train_p95)}</td><td class="num">${pct(s.mean_frr_at_train_p95)}</td>
        <td class="num">${pct(get(s, "operating_points.global_far_1pct.frr"))}</td>`;
      body.appendChild(tr);
    });

    const ts = DATA.time_split;
    const cmu = $("#cmuChecks");
    const cmuChecks = [
      ["Sessions 5 to 8 absent from training", ts.cmu.leakage_checks.sessions_5_8_absent_from_train],
      ["Every training session precedes every genuine test session", ts.cmu.leakage_checks.training_sessions_precede_genuine_test],
      ["Impostor attempts per user", 250],
    ];
    cmuChecks.forEach(([label, v]) => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${v === true ? "✓" : v}</b>${label}`;
      cmu.appendChild(li);
    });
    const per = Object.values(ts.balabit.per_user);
    const sum = (k) => per.reduce((a, r) => a + r[k], 0);
    const bal = $("#balabitChecks");
    const checks = ts.balabit.leakage_checks;
    [
      ["Fitting reads training files only", checks.training_files_only],
      ["Every enrollment action starts and ends before the cut", checks.enrollment_starts_before_cut && checks.enrollment_ends_at_or_before_cut],
      [`Latest enrollment action starts at ${fmt(ts.balabit.max_enrollment_action_start_fraction, 4)} of the session`, checks.max_enrollment_action_start_fraction_below_cut],
      [`${sum("enrollment_actions").toLocaleString()} enrollment actions, ${sum("holdout_actions").toLocaleString()} hold-out actions, ${sum("dropped_straddlers")} dropped at the cut`, true],
    ].forEach(([label, v]) => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${v ? "✓" : "✗"}</b>${label}`;
      bal.appendChild(li);
    });

    const dt = $("#driftTable tbody");
    [["Keystroke", DATA.drift.keystroke], ["Mouse", DATA.drift.mouse]].forEach(([label, d]) => {
      const s = d.threshold_summaries["0.2"];
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${label}</td><td class="num">${d.user_count}</td><td class="num">${pct(s.false_drift_rate)}</td><td class="num">${pct(s.pure_impostor_detection_rate)}</td>`;
      dt.appendChild(tr);
    });

    const op = $("#opTable tbody");
    const sm = DATA.keystroke.summary.scaled_manhattan;
    [
      ["Keystroke, scaled Manhattan", "user enrollment p95", sm.mean_far_at_train_p95, sm.mean_frr_at_train_p95],
      ["Keystroke, scaled Manhattan", "global FAR 1%", get(sm, "operating_points.global_far_1pct.far"), get(sm, "operating_points.global_far_1pct.frr")],
      ["Mouse", "user time hold-out p95", DATA.mouse.fixed_train_time_holdout_p95.far, DATA.mouse.fixed_train_time_holdout_p95.frr],
      ["Mouse", "global FAR 5%", get(DATA.mouse, "operating_points.global_far_5pct.far"), get(DATA.mouse, "operating_points.global_far_5pct.frr")],
      ["Session risk", "global FAR 1%", get(DATA.risk, "operating_points.global_far_1pct.far"), get(DATA.risk, "operating_points.global_far_1pct.frr")],
      ["Session risk", "global FAR 5%", get(DATA.risk, "operating_points.global_far_5pct.far"), get(DATA.risk, "operating_points.global_far_5pct.frr")],
    ].forEach(([model, thr, far, frr]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${model}</td><td class="muted">${thr}</td><td class="num">${pct(far)}</td><td class="num">${pct(frr)}</td>`;
      op.appendChild(tr);
    });
  }

  /* =========================================================
     Drift replay
     ========================================================= */
  function driftSeries(channel) {
    const byUser = {};
    if (channel === "keystroke") {
      const boundary = DATA.drift.keystroke.genuine_requests_per_user;
      DATA.drift.keystroke_points.forEach(([user, x, psi]) => {
        (byUser[user] = byUser[user] || []).push({ x, psi, impostor: x > boundary });
      });
    } else {
      DATA.drift.mouse_points.forEach(([user, x, phase, psi]) => {
        (byUser[user] = byUser[user] || []).push({ x, psi, impostor: phase === "i" });
      });
    }
    return byUser;
  }

  function setupDrift() {
    document.querySelectorAll("[data-ch]").forEach((b) => {
      b.addEventListener("click", () => {
        state.channel = b.dataset.ch;
        document.querySelectorAll("[data-ch]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        state.driftUser = null;
        populateDriftUsers();
        drawDrift();
      });
    });
    $("#driftUser").addEventListener("change", (e) => { state.driftUser = e.target.value; drawDrift(); });
    populateDriftUsers();
  }

  function populateDriftUsers() {
    const select = $("#driftUser");
    select.innerHTML = "";
    const users = Object.keys(driftSeries(state.channel)).sort();
    users.forEach((u) => {
      const opt = document.createElement("option");
      opt.value = u;
      opt.textContent = u;
      select.appendChild(opt);
    });
    state.driftUser = state.driftUser || users[0];
    select.value = state.driftUser;
  }

  function drawDrift() {
    const svg = $("#driftChart");
    clear(svg);
    const W = 1100, H = 300, L = 54, R = 14, T = 14, B = 40;
    const series = driftSeries(state.channel);
    const all = Object.values(series).flat();
    const xMax = Math.max(...all.map((p) => p.x));
    const xMin = Math.min(...all.map((p) => p.x));
    const lo = 1e-3, hi = 20;
    const x = (v) => L + ((v - xMin) / Math.max(xMax - xMin, 1)) * (W - L - R);
    const y = (v) => H - B - ((Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - T - B);
    [0.001, 0.01, 0.1, 1, 10].forEach((t) => {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: "gridline" }, svg);
      el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, svg, String(t));
    });
    Object.entries(series).forEach(([user, pts]) => {
      if (user === state.driftUser) return;
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.psi)}`).join(" ");
      el("path", { d, fill: "none", stroke: css("--ink-3"), "stroke-width": 0.8, opacity: 0.35 }, svg);
    });
    const threshold = DATA.drift.psi_threshold;
    el("line", { x1: L, x2: W - R, y1: y(threshold), y2: y(threshold), stroke: css("--accent"), "stroke-dasharray": "6 4", "stroke-width": 1.4 }, svg);
    el("text", { x: W - R, y: y(threshold) - 6, "text-anchor": "end", fill: css("--accent") }, svg, "PSI 0.2");
    const pts = series[state.driftUser] || [];
    const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.psi)}`).join(" ");
    el("path", { d, fill: "none", stroke: css("--ink"), "stroke-width": 1.6 }, svg);
    pts.forEach((p) => {
      el("circle", { cx: x(p.x), cy: y(p.psi), r: 3.4, fill: p.impostor ? css("--impostor") : css("--genuine") }, svg);
    });
    el("text", { x: (W + L) / 2, y: H - 8, "text-anchor": "middle" }, svg,
      state.channel === "keystroke" ? "keystroke requests replayed (first 200 genuine, then 250 impostor)" : "mouse sessions replayed (genuine sessions first, then impostor sessions)");
    el("text", { x: L - 46, y: T + 2 }, svg, "PSI");
    const firstFlag = pts.find((p) => p.psi > threshold);
    const genuineFlags = pts.filter((p) => !p.impostor && p.psi > threshold).length;
    $("#driftSummary").textContent = firstFlag
      ? `${state.driftUser}: first flag after ${firstFlag.x} ${state.channel === "keystroke" ? "requests" : "sessions"} (${firstFlag.impostor ? "impostor traffic" : "genuine traffic"})${genuineFlags ? `, ${genuineFlags} genuine checkpoint${genuineFlags > 1 ? "s" : ""} flagged` : ""}`
      : `${state.driftUser}: never crossed 0.2`;
  }

  function drawAging() {
    const svg = $("#agingChart");
    clear(svg);
    const W = 560, H = 220, L = 40, R = 10, T = 16, B = 34;
    const rows = Object.entries(DATA.drift.aging).sort((a, b) => Number(a[0]) - Number(b[0]));
    const lo = 0.8, hi = 1.2;
    const y = (v) => H - B - ((v - lo) / (hi - lo)) * (H - T - B);
    axisY(svg, y, [0.8, 0.9, 1.0, 1.1, 1.2], L, W, R, (t) => t.toFixed(1));
    el("line", { x1: L, x2: W - R, y1: y(1), y2: y(1), stroke: css("--ink-3"), "stroke-width": 1.2 }, svg);
    const band = (W - L - R) / rows.length;
    rows.forEach(([session, ratio], i) => {
      const x0 = L + band * i + band * 0.2;
      const w = band * 0.6;
      const color = Number(session) <= 4 ? css("--genuine") : css("--accent");
      const top = Math.min(y(ratio), y(1));
      el("rect", { x: x0, y: top, width: w, height: Math.abs(y(ratio) - y(1)) || 1, fill: color, opacity: 0.8, rx: 2 }, svg);
      el("text", { x: x0 + w / 2, y: ratio >= 1 ? y(ratio) - 6 : y(ratio) + 14, "text-anchor": "middle", fill: css("--ink") }, svg, ratio.toFixed(2));
      el("text", { x: x0 + w / 2, y: H - 12, "text-anchor": "middle" }, svg, `S${session}`);
    });
  }

  /* =========================================================
     Mouse pad: same segmentation and features as kma.mouse
     ========================================================= */
  const pad = { events: [], canvas: null, ctx: null, down: false };

  function setupPad() {
    pad.canvas = $("#padCanvas");
    pad.ctx = pad.canvas.getContext("2d");
    const area = $("#pad");
    const push = (e, state) => {
      const rect = pad.canvas.getBoundingClientRect();
      pad.events.push({ t: e.timeStamp / 1000, x: e.clientX - rect.left, y: e.clientY - rect.top, state });
      if (pad.events.length > 4000) pad.events.splice(0, pad.events.length - 4000);
      $("#padHint").style.display = "none";
    };
    area.addEventListener("pointermove", (e) => push(e, pad.down ? "Drag" : "Move"));
    area.addEventListener("pointerdown", (e) => { pad.down = true; area.setPointerCapture(e.pointerId); push(e, "Pressed"); });
    area.addEventListener("pointerup", (e) => { pad.down = false; push(e, "Released"); });
    $("#padClear").addEventListener("click", () => {
      pad.events = [];
      $("#padHint").style.display = "";
      drawPad();
    });
    (function loop() { drawPad(); requestAnimationFrame(loop); })();
  }

  function segment(events) {
    const segments = [];
    let start = 0;
    for (let i = 1; i < events.length; i++) {
      if (events[i].t - events[i - 1].t > 10 || events[i - 1].state === "Released") {
        segments.push(events.slice(start, i));
        start = i;
      }
    }
    if (events.length) segments.push(events.slice(start));
    return segments.filter((s) => s.length >= 4);
  }

  function actionFeatures(seg) {
    const speeds = [], curvatures = [];
    let prevAngle = null;
    for (let i = 1; i < seg.length; i++) {
      const dt = seg[i].t - seg[i - 1].t;
      if (dt <= 1e-6) continue;
      const dx = seg[i].x - seg[i - 1].x, dy = seg[i].y - seg[i - 1].y;
      const step = Math.hypot(dx, dy);
      speeds.push(step / dt);
      const angle = Math.atan2(dy / dt, dx / dt);
      if (prevAngle !== null && step > 0) {
        let delta = angle - prevAngle;
        while (delta > Math.PI) delta -= 2 * Math.PI;
        while (delta < -Math.PI) delta += 2 * Math.PI;
        curvatures.push(Math.abs(delta) / step);
      }
      prevAngle = angle;
    }
    const pressed = seg.findIndex((e) => e.state === "Pressed");
    const released = seg.findIndex((e, i) => i > pressed && e.state === "Released");
    const click = pressed >= 0 && released > pressed ? (seg[released].t - seg[pressed].t) * 1000 : null;
    const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
    return { speed: mean(speeds), curvature: mean(curvatures), click };
  }

  function drawPad() {
    if (!pad.canvas) return;
    const c = pad.canvas;
    const ratio = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * ratio) || c.height !== Math.round(h * ratio)) {
      c.width = Math.round(w * ratio);
      c.height = Math.round(h * ratio);
    }
    const ctx = pad.ctx;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const ev = pad.events;
    const slow = css("--genuine"), fast = css("--impostor"), mid = css("--accent");
    for (let i = 1; i < ev.length; i++) {
      const dt = ev[i].t - ev[i - 1].t;
      if (dt <= 0 || dt > 10 || ev[i - 1].state === "Released") continue;
      const speed = Math.hypot(ev[i].x - ev[i - 1].x, ev[i].y - ev[i - 1].y) / dt;
      ctx.strokeStyle = speed < 400 ? slow : speed < 1500 ? mid : fast;
      ctx.lineWidth = ev[i].state === "Drag" ? 3 : 1.6;
      ctx.globalAlpha = 0.25 + 0.75 * (i / ev.length);
      ctx.beginPath();
      ctx.moveTo(ev[i - 1].x, ev[i - 1].y);
      ctx.lineTo(ev[i].x, ev[i].y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ev.forEach((e) => {
      if (e.state === "Pressed") {
        ctx.fillStyle = mid;
        ctx.beginPath();
        ctx.arc(e.x, e.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    const actions = segment(ev).map(actionFeatures);
    const avg = (key) => {
      const v = actions.map((a) => a[key]).filter((x) => x != null);
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
    };
    $("#sActions").textContent = actions.length;
    $("#sSpeed").textContent = avg("speed") == null ? "-" : Math.round(avg("speed"));
    $("#sCurv").textContent = avg("curvature") == null ? "-" : avg("curvature").toFixed(3);
    $("#sClick").textContent = avg("click") == null ? "-" : Math.round(avg("click"));
  }
})();
