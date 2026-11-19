/* ============================================================
   MLP 学习参数可视化
   2 → h → 1，sigmoid + 二元交叉熵，手写前向 / 反向传播
   ============================================================ */
(() => {
  "use strict";

  /* ---------------- 数据与数学工具 ---------------- */
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const T = [0, 1, 1, 0];
  const EPS = 1e-9;
  const SEED = 20261119;

  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const f2 = v => (Number.isFinite(v) ? v.toFixed(2) : "—");

  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------------- 模型 ---------------- */
  /* 参数打包成一个数组：
     0 .. 2h-1   W1 (输入 → 隐藏)，索引 i*h + j
     2h .. 3h-1  b1
     3h .. 4h-1  W2 (隐藏 → 输出)
     4h          b2                                             */
  class MLP {
    constructor(h, scale) {
      this.h = h;
      this.p = new Float64Array(4 * h + 1);
      const rnd = mulberry32(SEED);
      for (let i = 0; i < this.p.length; i++) this.p[i] = (rnd() * 2 - 1) * scale;
      this.epoch = 0;
      this.m = new Float64Array(this.p.length);
      this.v = new Float64Array(this.p.length);
      this.t = 0;
    }

    forward(x) {
      const h = this.h, p = this.p;
      const z1 = new Array(h), a = new Array(h);
      for (let j = 0; j < h; j++) {
        z1[j] = x[0] * p[j] + x[1] * p[h + j] + p[2 * h + j];
        a[j] = sigmoid(z1[j]);
      }
      let z2 = p[4 * h];
      for (let j = 0; j < h; j++) z2 += a[j] * p[3 * h + j];
      return { z1, a, z2, pred: sigmoid(z2) };
    }

    /* 不分配数组的快速预测，用于决策边界 */
    predict(x0, x1) {
      const h = this.h, p = this.p;
      let z2 = p[4 * h];
      for (let j = 0; j < h; j++) {
        z2 += sigmoid(x0 * p[j] + x1 * p[h + j] + p[2 * h + j]) * p[3 * h + j];
      }
      return sigmoid(z2);
    }

    gradient() {
      const h = this.h, p = this.p, g = new Float64Array(p.length);
      let loss = 0;
      for (let n = 0; n < 4; n++) {
        const { a, pred } = this.forward(X[n]);
        const t = T[n];
        const pc = clamp(pred, EPS, 1 - EPS);
        loss += -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
        const dz2 = (pc - t) / 4;
        g[4 * h] += dz2;
        for (let j = 0; j < h; j++) {
          g[3 * h + j] += a[j] * dz2;
          const dz1 = dz2 * p[3 * h + j] * a[j] * (1 - a[j]);
          g[2 * h + j] += dz1;
          g[j] += X[n][0] * dz1;
          g[h + j] += X[n][1] * dz1;
        }
      }
      return { loss: loss / 4, g };
    }

    step(lr, opt) {
      const { loss, g } = this.gradient();
      if (!Number.isFinite(loss)) return { loss: NaN };
      this.t++;
      const p = this.p, n = p.length;
      if (opt === "adam") {
        for (let i = 0; i < n; i++) {
          this.m[i] = 0.9 * this.m[i] + 0.1 * g[i];
          this.v[i] = 0.999 * this.v[i] + 0.001 * g[i] * g[i];
          const mh = this.m[i] / (1 - Math.pow(0.9, this.t));
          const vh = this.v[i] / (1 - Math.pow(0.999, this.t));
          p[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
        }
      } else if (opt === "momentum") {
        for (let i = 0; i < n; i++) {
          this.m[i] = 0.9 * this.m[i] + g[i];
          p[i] -= lr * this.m[i];
        }
      } else {
        for (let i = 0; i < n; i++) p[i] -= lr * g[i];
      }
      this.epoch++;
      return { loss };
    }
  }

  /* ---------------- 状态 ---------------- */
  const state = {
    h: 2,
    lr: 3,
    scale: 1,
    opt: "gd",
    speed: 10,
    model: null,
    history: [],
    running: false,
    selected: 0,
    diverged: false,
    lastLoss: NaN
  };

  const $ = id => document.getElementById(id);
  const netSvg = $("netSvg");

  /* 学习率滑杆：0–100 映射到 0.01–20（对数刻度） */
  const LR_LO = -2, LR_HI = 1.3;
  const sliderToLr = v => Math.pow(10, LR_LO + (v / 100) * (LR_HI - LR_LO));
  const lrToSlider = lr =>
    Math.round(((Math.log10(lr) - LR_LO) / (LR_HI - LR_LO)) * 100);

  /* ---------------- 网络 SVG ---------------- */
  const NS = "http://www.w3.org/2000/svg";
  let netParts = null;

  function mk(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function buildNet() {
    const h = state.h;
    netSvg.innerHTML = "";

    const inX = 96, hidX = 280, outX = 470;
    const inY = [108, 212];
    const outY = 160;
    const hidY = h === 1 ? [160] : Array.from({ length: h }, (_, j) => 74 + j * (172 / (h - 1)));

    const gLines = mk("g");
    const gText = mk("g");
    const gNodes = mk("g");

    const lines = [];
    const labels = [];

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < h; j++) {
        const ln = mk("line", {
          x1: inX + 21, y1: inY[i], x2: hidX - 23, y2: hidY[j],
          "stroke-linecap": "round"
        });
        gLines.appendChild(ln);
        lines.push({ el: ln, idx: i * h + j, layer: 1 });

        if (h <= 3) {
          const tx = mk("text", {
            x: (inX + hidX) / 2 - 16, y: (inY[i] + hidY[j]) / 2 - 6,
            "font-size": "9.5", "text-anchor": "middle", fill: "#9096a8"
          });
          gText.appendChild(tx);
          labels.push({ el: tx, idx: i * h + j });
        }
      }
    }

    for (let j = 0; j < h; j++) {
      const ln = mk("line", {
        x1: hidX + 23, y1: hidY[j], x2: outX - 21, y2: outY,
        "stroke-linecap": "round"
      });
      gLines.appendChild(ln);
      lines.push({ el: ln, idx: 3 * h + j, layer: 2 });

      if (h <= 3) {
        const tx = mk("text", {
          x: (hidX + outX) / 2 + 8, y: (hidY[j] + outY) / 2 - 6,
          "font-size": "9.5", "text-anchor": "middle", fill: "#9096a8"
        });
        gText.appendChild(tx);
        labels.push({ el: tx, idx: 3 * h + j });
      }
    }

    const inNodes = [], hidNodes = [];
    for (let i = 0; i < 2; i++) {
      const c = mk("circle", {
        cx: inX, cy: inY[i], r: 20, fill: "rgba(255,255,255,0.05)",
        stroke: "rgba(255,255,255,0.18)", "stroke-width": 1.5
      });
      const v = mk("text", {
        x: inX, y: inY[i] + 4, "text-anchor": "middle",
        "font-size": "10", fill: "#d7dbe8", "font-family": "ui-monospace, Consolas, monospace"
      });
      const lab = mk("text", {
        x: inX, y: inY[i] + 40, "text-anchor": "middle",
        "font-size": "10.5", fill: "#6d7488", "font-family": "ui-monospace, Consolas, monospace"
      });
      lab.textContent = i === 0 ? "x₁" : "x₂";
      v.textContent = "0";
      gNodes.append(c, v, lab);
      inNodes.push({ c, v });
    }

    for (let j = 0; j < h; j++) {
      const c = mk("circle", {
        cx: hidX, cy: hidY[j], r: 21, fill: "rgba(124,92,255,0.12)",
        stroke: "rgba(167,139,250,0.45)", "stroke-width": 1.5
      });
      const v = mk("text", {
        x: hidX, y: hidY[j] + 4, "text-anchor": "middle",
        "font-size": "9.5", fill: "#d7dbe8", "font-family": "ui-monospace, Consolas, monospace"
      });
      const lab = mk("text", {
        x: hidX, y: hidY[j] + 41, "text-anchor": "middle",
        "font-size": "10.5", fill: "#6d7488", "font-family": "ui-monospace, Consolas, monospace"
      });
      lab.textContent = "h" + (j + 1);
      v.textContent = "0";
      gNodes.append(c, v, lab);
      hidNodes.push({ c, v });
    }

    const outC = mk("circle", {
      cx: outX, cy: outY, r: 23, fill: "rgba(34,211,238,0.1)",
      stroke: "rgba(34,211,238,0.45)", "stroke-width": 1.5
    });
    const outV = mk("text", {
      x: outX, y: outY + 4, "text-anchor": "middle",
      "font-size": "10", fill: "#d7dbe8", "font-family": "ui-monospace, Consolas, monospace"
    });
    const outLab = mk("text", {
      x: outX, y: outY + 43, "text-anchor": "middle",
      "font-size": "10.5", fill: "#6d7488", "font-family": "ui-monospace, Consolas, monospace"
    });
    outLab.textContent = "ŷ";
    outV.textContent = "0";
    gNodes.append(outC, outV, outLab);

    netSvg.append(gLines, gText, gNodes);
    netParts = { lines, labels, inNodes, hidNodes, outV, hidY, inY };
    $("netBadge").textContent = `2 → ${h} → 1`;
  }

  function updateNet() {
    if (!netParts) return;
    const m = state.model;
    const p = m.p;
    const out = m.forward(X[state.selected]);

    let wmax = 1e-6;
    for (let i = 0; i < p.length; i++) wmax = Math.max(wmax, Math.abs(p[i]));

    for (const L of netParts.lines) {
      const w = p[L.idx];
      const strength = Math.min(1, Math.abs(w) / wmax);
      L.el.setAttribute("stroke", w >= 0 ? "#5b8cff" : "#fb923c");
      L.el.setAttribute("stroke-width", (1 + 8 * strength).toFixed(2));
      L.el.setAttribute("opacity", (0.3 + 0.6 * strength).toFixed(2));
    }

    for (const L of netParts.labels) {
      L.el.textContent = f2(p[L.idx]);
    }

    netParts.inNodes.forEach((n, i) => {
      n.v.textContent = X[state.selected][i];
      n.c.setAttribute("stroke", i === 0 ? "rgba(34,211,238,0.7)" : "rgba(34,211,238,0.35)");
      n.c.setAttribute("fill", "rgba(34,211,238,0.07)");
    });

    netParts.hidNodes.forEach((n, j) => {
      const a = out.a[j];
      n.v.textContent = a.toFixed(2);
      n.c.setAttribute("fill", `rgba(124,92,255,${(0.08 + 0.5 * a).toFixed(3)})`);
      n.c.setAttribute("stroke", `rgba(167,139,250,${(0.3 + 0.6 * a).toFixed(3)})`);
    });

    netParts.outV.textContent = out.pred.toFixed(2);
    netParts.outV.setAttribute("fill", out.pred >= 0.5 ? "#fb923c" : "#7aa2ff");
  }

  /* ---------------- 样本列表 ---------------- */
  const sampleRefs = [];

  function buildSamples() {
    const box = $("sampleList");
    box.innerHTML = "";
    X.forEach((x, n) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-row";
      btn.innerHTML =
        `<span class="sx">[${x[0]}, ${x[1]}]</span>` +
        `<span class="st">→ 目标 ${T[n]}</span>` +
        `<span class="sp">—</span>`;
      btn.addEventListener("click", () => {
        state.selected = n;
        sampleRefs.forEach((r, i) => r.btn.classList.toggle("active", i === n));
        render(true);
      });
      box.appendChild(btn);
      sampleRefs.push({ btn, sp: btn.querySelector(".sp") });
    });
    sampleRefs[0].btn.classList.add("active");
  }

  function updateSamples() {
    const m = state.model;
    X.forEach((x, n) => {
      const p = m.forward(x).pred;
      const ok = (p >= 0.5 ? 1 : 0) === T[n];
      const ref = sampleRefs[n];
      ref.sp.textContent = `P(1) = ${(p * 100).toFixed(1)}%`;
      ref.sp.classList.toggle("bad", !ok);
    });

    const p = m.forward(X[state.selected]).pred;
    const pred = p >= 0.5 ? 1 : 0;
    const ok = pred === T[state.selected];
    $("sampleNote").innerHTML =
      `阈值 0.5 → 判为 <b>${pred}</b>，目标 <b>${T[state.selected]}</b> · ` +
      (ok ? '<span style="color:#34d399">正确</span>' : '<span style="color:#f87171">错误</span>');
  }

  /* ---------------- 推导步骤 ---------------- */
  function renderSteps() {
    const m = state.model;
    const h = m.h;
    const x = X[state.selected];
    const out = m.forward(x);
    const p = m.p;

    $("stepTag").textContent =
      `样本 [${x[0]}, ${x[1]}] · 目标 ${T[state.selected]}`;

    let zLines = "";
    for (let j = 0; j < h; j++) {
      zLines +=
        `<div>z<sub>${j + 1}</sub> = ${x[0]} × ${f2(p[j])} + ${x[1]} × ${f2(p[h + j])} + (${f2(p[2 * h + j])})<br>` +
        `= <b>${f2(out.z1[j])}</b></div>`;
    }

    let hLines = "";
    for (let j = 0; j < h; j++) {
      hLines += `<div>h<sub>${j + 1}</sub> = σ(${f2(out.z1[j])}) = <b class="hl">${out.a[j].toFixed(3)}</b></div>`;
    }

    let logit = `logit = `;
    const terms = [];
    for (let j = 0; j < h; j++) terms.push(`${out.a[j].toFixed(3)} × ${f2(p[3 * h + j])}`);
    logit += terms.join(" + ") + ` + (${f2(p[4 * h])})`;
    logit += `<br>= <b>${f2(out.z2)}</b>`;

    const prob = out.pred;
    const pred = prob >= 0.5 ? 1 : 0;
    const ok = pred === T[state.selected];

    $("steps").innerHTML = `
      <div class="step-item">
        <span class="sn">STEP 01</span>
        <h4>隐藏层的线性组合 z = xW + b</h4>
        <div class="step-eq">${zLines}</div>
      </div>
      <div class="step-item">
        <span class="sn">STEP 02</span>
        <h4>过 Sigmoid 激活 h = σ(z)</h4>
        <div class="step-eq">${hLines}</div>
      </div>
      <div class="step-item">
        <span class="sn">STEP 03</span>
        <h4>输出层 ŷ = σ(hv + c)</h4>
        <div class="step-eq">${logit}</div>
      </div>
      <div class="step-item">
        <span class="sn">STEP 04</span>
        <h4>变成概率，再按 0.5 判定</h4>
        <div class="step-eq">p = σ(${f2(out.z2)}) = <b class="hl">${(prob * 100).toFixed(2)}%</b></div>
        <p class="step-verdict">
          阈值 0.5 → 预测 <strong>${pred}</strong>，目标 <strong>${T[state.selected]}</strong> ·
          ${ok ? '<span class="ok">分类正确</span>' : '<span class="no">分类错误</span>'}
        </p>
      </div>`;
  }

  /* ---------------- 指标 ---------------- */
  function accuracy() {
    let ok = 0;
    for (let n = 0; n < 4; n++) {
      if ((state.model.predict(X[n][0], X[n][1]) >= 0.5 ? 1 : 0) === T[n]) ok++;
    }
    return ok / 4;
  }

  function updateMetrics() {
    const m = state.model;
    $("mEpoch").textContent = m.epoch;
    $("mLoss").textContent = state.diverged ? "NaN" : state.lastLoss.toFixed(4);
    $("mAcc").innerHTML = `${Math.round(accuracy() * 100)}<em>%</em>`;

    let wmax = 0;
    for (let i = 0; i < m.p.length; i++) wmax = Math.max(wmax, Math.abs(m.p[i]));
    $("mW").textContent = state.diverged ? "NaN" : wmax.toFixed(2);
  }

  /* ---------------- 损失曲线 ---------------- */
  function drawLoss() {
    const cv = $("lossChart");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 64, padR = 26, padT = 24, padB = 48;
    const iw = W - padL - padR, ih = H - padT - padB;
    const hist = state.history;
    const maxE = Math.max(300, hist[hist.length - 1].e);

    let maxL = 0.75;
    for (const d of hist) if (Number.isFinite(d.l)) maxL = Math.max(maxL, d.l);
    maxL = Math.min(maxL * 1.12, 6);

    // 网格 + 纵轴刻度
    ctx.font = "20px ui-monospace, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let k = 0; k <= 4; k++) {
      const y = padT + ih * (k / 4);
      const val = maxL * (1 - k / 4);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      ctx.fillStyle = "#5e667a";
      ctx.fillText(val.toFixed(2), padL - 12, y);
    }

    // 横轴刻度
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let k = 0; k <= 3; k++) {
      const e = Math.round((maxE * k) / 3);
      const x = padL + iw * (e / maxE);
      ctx.fillStyle = "#5e667a";
      ctx.fillText(String(e), x, H - padB + 14);
    }

    // 曲线
    if (hist.length > 1) {
      const grad = ctx.createLinearGradient(padL, 0, W - padR, 0);
      grad.addColorStop(0, "#7c5cff");
      grad.addColorStop(1, "#22d3ee");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      for (const d of hist) {
        if (!Number.isFinite(d.l)) continue;
        const x = padL + iw * (d.e / maxE);
        const y = padT + ih * (1 - clamp(d.l, 0, maxL) / maxL);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const last = hist[hist.length - 1];
      if (Number.isFinite(last.l)) {
        const lx = padL + iw * (last.e / maxE);
        const ly = padT + ih * (1 - clamp(last.l, 0, maxL) / maxL);
        ctx.fillStyle = "#22d3ee";
        ctx.shadowColor = "rgba(34,211,238,0.9)";
        ctx.shadowBlur = 14;
        ctx.beginPath();
        ctx.arc(lx, ly, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // 轴标签
    ctx.fillStyle = "#6d7488";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = "20px ui-monospace, Consolas, monospace";
    ctx.fillText("loss", padL - 4, 2);
    ctx.textAlign = "right";
    ctx.fillText("epoch", W - padR, H - 30);
  }

  /* ---------------- 决策边界 ---------------- */
  function drawBoundary() {
    const cv = $("boundary");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    const m = 52;
    const grid = 56;
    const cell = (W - 2 * m) / grid;
    const m2 = state.model;

    ctx.clearRect(0, 0, W, H);

    // 概率热力
    const cls = new Array(grid * grid);
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const x0 = -0.35 + ((gx + 0.5) / grid) * 1.7;
        const x1 = 1.35 - ((gy + 0.5) / grid) * 1.7;
        const p = m2.predict(x0, x1);
        const isOne = p >= 0.5;
        cls[gy * grid + gx] = isOne ? 1 : 0;
        const s = Math.min(1, Math.abs(p - 0.5) * 2);
        ctx.fillStyle = isOne
          ? `rgba(251,146,60,${(0.05 + 0.5 * s).toFixed(3)})`
          : `rgba(91,140,255,${(0.05 + 0.5 * s).toFixed(3)})`;
        ctx.fillRect(m + gx * cell, m + gy * cell, cell + 1, cell + 1);
      }
    }

    // 0.5 等值线（按格点边界近似）
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const c = cls[gy * grid + gx];
        if (gx < grid - 1 && cls[gy * grid + gx + 1] !== c) {
          const x = m + (gx + 1) * cell;
          const y = m + gy * cell;
          ctx.moveTo(x, y);
          ctx.lineTo(x, y + cell);
        }
        if (gy < grid - 1 && cls[(gy + 1) * grid + gx] !== c) {
          const x = m + gx * cell;
          const y = m + (gy + 1) * cell;
          ctx.moveTo(x, y);
          ctx.lineTo(x + cell, y);
        }
      }
    }
    ctx.stroke();

    // 坐标框
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(m, m, W - 2 * m, H - 2 * m);

    // 四个样本点
    ctx.font = "22px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    X.forEach((x, n) => {
      const px = m + (x[0] + 0.35) / 1.7 * (W - 2 * m);
      const py = m + (1 - (x[1] + 0.35) / 1.7) * (H - 2 * m);
      const p = m2.predict(x[0], x[1]);
      const wrong = (p >= 0.5 ? 1 : 0) !== T[n];

      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI * 2);
      ctx.fillStyle = T[n] === 1 ? "#fb923c" : "#5b8cff";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = wrong ? "#f87171" : "rgba(255,255,255,0.9)";
      ctx.stroke();

      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillText(`${x[0]},${x[1]}`, px + (x[0] === 0 ? 38 : -38), py + (x[1] === 0 ? 28 : -28));
    });

    // 坐标标签
    ctx.fillStyle = "#6d7488";
    ctx.font = "20px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("x₁", W / 2, H - 32);
    ctx.save();
    ctx.translate(20, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("x₂", 0, 0);
    ctx.restore();
  }

  /* ---------------- 参数表 ---------------- */
  function renderParams() {
    const m = state.model;
    const h = m.h;
    const { g } = m.gradient();
    const body = $("paramBody");

    const gcell = v => {
      const cls = v >= 0 ? "g-pos" : "g-neg";
      return `<td class="${cls}">${v >= 0 ? "+" : ""}${v.toFixed(4)}</td>`;
    };

    let html = `<tr class="group-row"><td colspan="4">输入层 → 隐藏层 · W1 (2 × ${h})</td></tr>`;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < h; j++) {
        const idx = i * h + j;
        html += `<tr><td>W1[x${i + 1}→h${j + 1}]</td><td>x${i + 1} 到 h${j + 1}</td>` +
          `<td>${m.p[idx].toFixed(4)}</td>${gcell(g[idx])}</tr>`;
      }
    }

    html += `<tr class="group-row"><td colspan="4">隐藏层偏置 · b1 (${h})</td></tr>`;
    for (let j = 0; j < h; j++) {
      const idx = 2 * h + j;
      html += `<tr><td>b1[h${j + 1}]</td><td>h${j + 1} 的偏置</td>` +
        `<td>${m.p[idx].toFixed(4)}</td>${gcell(g[idx])}</tr>`;
    }

    html += `<tr class="group-row"><td colspan="4">隐藏层 → 输出层 · W2 (${h} × 1)</td></tr>`;
    for (let j = 0; j < h; j++) {
      const idx = 3 * h + j;
      html += `<tr><td>W2[h${j + 1}→ŷ]</td><td>h${j + 1} 到输出</td>` +
        `<td>${m.p[idx].toFixed(4)}</td>${gcell(g[idx])}</tr>`;
    }

    html += `<tr class="group-row"><td colspan="4">输出层偏置 · b2 (1)</td></tr>`;
    html += `<tr><td>b2[ŷ]</td><td>输出层偏置</td>` +
      `<td>${m.p[4 * h].toFixed(4)}</td>${gcell(g[4 * h])}</tr>`;

    body.innerHTML = html;
    $("paramCount").textContent = m.p.length;
  }

  /* ---------------- 训练循环 ---------------- */
  let raf = null;
  let lastBoundary = 0;
  let lastParams = 0;

  function record(loss) {
    const e = state.model.epoch;
    const hist = state.history;
    if (hist.length < 3000 || e % 5 === 0) hist.push({ e, l: loss });
  }

  function stepOnce() {
    const res = state.model.step(state.lr, state.opt);
    if (!Number.isFinite(res.loss) || !Number.isFinite(state.model.p[0])) {
      state.diverged = true;
      stopTraining();
      return;
    }
    state.lastLoss = res.loss;
    record(res.loss);
    if (res.loss < 0.003 || state.model.epoch >= 8000) stopTraining();
  }

  function loop() {
    if (!state.running) return;
    for (let i = 0; i < state.speed; i++) {
      if (!state.running) break;
      stepOnce();
    }
    render();
    if (state.running) raf = requestAnimationFrame(loop);
  }

  function startTraining() {
    if (state.diverged) reset(false);
    state.running = true;
    $("playBtn").textContent = "暂停";
    $("playBtn").classList.add("running");
    raf = requestAnimationFrame(loop);
  }

  function stopTraining() {
    state.running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    const btn = $("playBtn");
    btn.textContent = state.model.epoch === 0 ? "开始训练" : "继续训练";
    btn.classList.remove("running");
  }

  function render(force = false) {
    updateMetrics();
    updateNet();
    updateSamples();
    renderSteps();
    drawLoss();
    const now = performance.now();
    if (force || now - lastBoundary > 130) {
      drawBoundary();
      lastBoundary = now;
    }
    if (force || now - lastParams > 170) {
      renderParams();
      lastParams = now;
    }
  }

  function reset(rebuild) {
    stopTraining();
    state.model = new MLP(state.h, state.scale);
    state.history = [];
    state.diverged = false;
    const initLoss = state.model.gradient().loss;
    state.lastLoss = initLoss;
    state.history.push({ e: 0, l: initLoss });
    if (rebuild) buildNet();
    $("playBtn").textContent = "开始训练";
    render(true);
  }

  /* ---------------- 学习率对比实验 ---------------- */
  function runCompare() {
    const lrs = [0.1, 3, 15];
    const colors = ["#22d3ee", "#a78bfa", "#fb923c"];
    const EPOCHS = 300;

    const runs = lrs.map((lr, k) => {
      const m = new MLP(state.h, state.scale);
      const pts = [{ e: 0, l: m.gradient().loss }];
      let diverged = false;
      for (let e = 1; e <= EPOCHS; e++) {
        const r = m.step(lr, "gd");
        if (!Number.isFinite(r.loss) || !Number.isFinite(m.p[0])) {
          diverged = true;
          break;
        }
        pts.push({ e, l: r.loss });
      }
      return { lr, pts, color: colors[k], diverged, final: pts[pts.length - 1].l };
    });

    drawCompare(runs);

    $("compareLegend").innerHTML = runs
      .map(r => {
        const tail = r.diverged
          ? '<span style="color:#f87171">发散</span>'
          : `末轮 loss ${r.final.toFixed(4)}`;
        return `<li><i style="background:${r.color}"></i>lr = ${r.lr} · ${tail}</li>`;
      })
      .join("");
  }

  function drawCompare(runs) {
    const cv = $("compareChart");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 64, padR = 26, padT = 24, padB = 48;
    const iw = W - padL - padR, ih = H - padT - padB;
    const maxE = 300;

    let maxL = 0.8;
    for (const r of runs) {
      for (const d of r.pts) if (Number.isFinite(d.l)) maxL = Math.max(maxL, d.l);
    }
    maxL = Math.min(maxL * 1.1, 4);

    ctx.font = "20px ui-monospace, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let k = 0; k <= 4; k++) {
      const y = padT + ih * (k / 4);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      ctx.fillStyle = "#5e667a";
      ctx.fillText((maxL * (1 - k / 4)).toFixed(2), padL - 12, y);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let k = 0; k <= 3; k++) {
      const e = Math.round((maxE * k) / 3);
      ctx.fillStyle = "#5e667a";
      ctx.fillText(String(e), padL + iw * (e / maxE), H - padB + 14);
    }

    for (const r of runs) {
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      for (const d of r.pts) {
        if (!Number.isFinite(d.l)) continue;
        const x = padL + iw * (d.e / maxE);
        const y = padT + ih * (1 - clamp(d.l, 0, maxL) / maxL);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.fillStyle = "#6d7488";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("loss", padL - 4, 2);
    ctx.textAlign = "right";
    ctx.fillText("epoch", W - padR, H - 30);
  }

  /* ---------------- 交互绑定 ---------------- */
  function bindUI() {
    const lr = $("lr"), lrOut = $("lrOut");

    lr.addEventListener("input", () => {
      state.lr = sliderToLr(Number(lr.value));
      lrOut.textContent = state.lr < 0.1 ? state.lr.toFixed(3) : state.lr.toFixed(2);
        $("lrHint").textContent =
        state.lr < 0.3 ? "偏小：收敛会很慢" :
        state.lr > 8 ? "偏大：容易震荡或发散" :
        "比较合适：能稳定下降";
    });

    $("units").addEventListener("change", e => {
      state.h = Number(e.target.value);
      reset(true);
    });

    $("scale").addEventListener("change", e => {
      state.scale = Number(e.target.value);
      reset(false);
    });

    $("opt").addEventListener("change", e => {
      state.opt = e.target.value;
      const preset = state.opt === "adam" ? 0.03 : state.opt === "momentum" ? 1 : 3;
      state.lr = preset;
      lr.value = lrToSlider(preset);
      lrOut.textContent = preset.toFixed(2);
      $("lrHint").textContent =
        state.opt === "adam" ? "Adam 建议 0.01 – 0.05" :
        state.opt === "momentum" ? "动量建议 0.3 – 1.5" :
        "GD 建议 0.5 – 5";
      reset(false);
    });

    $("speed").addEventListener("input", e => {
      state.speed = Number(e.target.value);
      $("speedOut").textContent = state.speed;
    });

    $("playBtn").addEventListener("click", () => {
      if (state.running) stopTraining();
      else startTraining();
    });

    $("stepBtn").addEventListener("click", () => {
      stopTraining();
      if (state.diverged) reset(false);
      stepOnce();
      render(true);
    });

    $("resetBtn").addEventListener("click", () => reset(false));

    $("compareBtn").addEventListener("click", runCompare);

    const navToggle = $("navToggle");
    const navLinks = $("navLinks");
    navToggle.addEventListener("click", () => {
      const open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    navLinks.querySelectorAll("a").forEach(a =>
      a.addEventListener("click", () => {
        navLinks.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
      })
    );

    $("year").textContent = new Date().getFullYear();
  }

  /* ---------------- 启动 ---------------- */
  function init() {
    state.model = new MLP(state.h, state.scale);
    state.lastLoss = state.model.gradient().loss;
    state.history.push({ e: 0, l: state.lastLoss });

    $("lr").value = lrToSlider(state.lr);
    $("lrOut").textContent = state.lr.toFixed(2);

    buildNet();
    buildSamples();
    bindUI();

    /* 滚动入场 */
    const io = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -6% 0px" }
    );
    document.querySelectorAll(".reveal").forEach(el => io.observe(el));

    render(true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
