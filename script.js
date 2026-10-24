/* ============================================================
   报告数据 —— 改成你自己的真实数据即可
   ============================================================ */
const REPORT = {
  kpis: [
    { label: "主力 Agent", value: 1, decimals: 0, suffix: " 个" },
    { label: "协作会话", value: 214, decimals: 0, suffix: " 次" },
    { label: "完成项目", value: 12, decimals: 0, suffix: " 个" },
    { label: "节省时间", value: 96, decimals: 0, suffix: " h" }
  ],
  agents: [
    {
      name: "从 0 到 1 建站",
      color: "#7c5cff",
      share: 34,
      sessions: 73,
      rating: 9.4,
      use: "需求 → 结构 → 页面 → 部署",
      note: "一句需求先立起项目骨架，再按页面逐块补齐。适合把模糊想法快速变成能打开的东西。"
    },
    {
      name: "改 Bug 与调试",
      color: "#22d3ee",
      share: 26,
      sessions: 56,
      rating: 9.0,
      use: "读报错 · 定位 · 最小改动",
      note: "把完整报错和相关代码一起给它，比只描述现象快得多；改完一定自己复现一遍。"
    },
    {
      name: "读代码 · 读懂报错",
      color: "#f472b6",
      share: 22,
      sessions: 47,
      rating: 8.6,
      use: "陌生仓库 · 调用链 · 概念解释",
      note: "接手别人的代码时先让它画出调用链，再逐段追问，比自己从头读快很多。"
    },
    {
      name: "文档与笔记",
      color: "#fbbf24",
      share: 18,
      sessions: 38,
      rating: 8.2,
      use: "README · 实验报告 · 知识整理",
      note: "把零散笔记丢给它整理成结构化文档，但结论和措辞要自己再过一遍。"
    }
  ],
  weekly: [14, 19, 23, 18, 27, 31, 38, 44]
};

const NS = "http://www.w3.org/2000/svg";
const RADIUS = 70;
const CIRC = 2 * Math.PI * RADIUS;
const GAP = 5;

/* ============================================================
   KPI
   ============================================================ */
function renderKpis() {
  const host = document.getElementById("kpis");
  host.innerHTML = REPORT.kpis
    .map(
      (k, i) => `
      <div class="kpi" style="transition-delay:${i * 60}ms">
        <div class="value"><span data-value="${k.value}" data-decimals="${k.decimals}">0</span><em>${k.suffix}</em></div>
        <div class="label">${k.label}</div>
      </div>`
    )
    .join("");
}

/* ============================================================
   环形图
   ============================================================ */
function renderDonut() {
  const host = document.getElementById("donutChart");
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 200 200");

  const base = document.createElementNS(NS, "circle");
  base.setAttribute("cx", 100);
  base.setAttribute("cy", 100);
  base.setAttribute("r", RADIUS);
  base.setAttribute("fill", "none");
  base.setAttribute("stroke", "rgba(255,255,255,0.06)");
  base.setAttribute("stroke-width", 18);
  svg.appendChild(base);

  let acc = 0;
  REPORT.agents.forEach((agent) => {
    const len = (agent.share / 100) * CIRC;
    const circle = document.createElementNS(NS, "circle");
    circle.setAttribute("cx", 100);
    circle.setAttribute("cy", 100);
    circle.setAttribute("r", RADIUS);
    circle.setAttribute("fill", "none");
    circle.setAttribute("stroke", agent.color);
    circle.setAttribute("stroke-width", 18);
    circle.setAttribute("stroke-linecap", "round");
    circle.setAttribute("transform", "rotate(-90 100 100)");
    circle.style.strokeDasharray = `0 ${CIRC}`;
    circle.style.strokeDashoffset = String(-acc);
    circle.style.transition = "stroke-dasharray 1s cubic-bezier(0.22, 1, 0.36, 1)";
    circle.dataset.dash = `${Math.max(len - GAP, 1)} ${CIRC}`;
    acc += len;
    svg.appendChild(circle);
  });

  host.appendChild(svg);

  document.getElementById("donutLegend").innerHTML = REPORT.agents
    .map(
      (a) => `
      <li>
        <span class="dot" style="background:${a.color}"></span>
        <span class="lname">${a.name}</span>
        <span class="lval">${a.share}%</span>
      </li>`
    )
    .join("");
}

/* ============================================================
   柱状图
   ============================================================ */
function renderBars() {
  const host = document.getElementById("barChart");
  const max = Math.max(...REPORT.weekly);

  host.innerHTML = REPORT.weekly
    .map(
      (value, i) => `
      <div class="bar" title="第 ${i + 1} 周 · ${value} 次会话">
        <span class="bar-value">${value}</span>
        <div class="bar-track"><i class="bar-fill" data-h="${Math.round((value / max) * 100)}"></i></div>
        <span class="bar-label">W${i + 1}</span>
      </div>`
    )
    .join("");
}

/* ============================================================
   Agent 卡片
   ============================================================ */
function renderAgents() {
  const host = document.getElementById("agentGrid");
  host.innerHTML = REPORT.agents
    .map(
      (a, i) => `
      <article class="agent-card card reveal" style="--c:${a.color};transition-delay:${i * 70}ms">
        <header>
          <span class="agent-rank">0${i + 1}</span>
          <h3>${a.name}</h3>
          <span class="agent-share">${a.share}%</span>
        </header>
        <p class="agent-use">${a.use}</p>
        <p class="agent-note">${a.note}</p>
        <div class="meter"><i style="width:${a.rating * 10}%"></i></div>
        <div class="agent-meta">
          <span>${a.sessions} 次会话</span>
          <span>顺手度 ${a.rating.toFixed(1)}</span>
        </div>
      </article>`
    )
    .join("");
}

/* ============================================================
   数字滚动
   ============================================================ */
function countUp(el) {
  const target = parseFloat(el.dataset.value);
  const decimals = parseInt(el.dataset.decimals || "0", 10);
  const duration = 1200;
  const start = performance.now();

  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * eased).toFixed(decimals);
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ============================================================
   图表入场动画
   ============================================================ */
function animateCharts() {
  const circles = document.querySelectorAll("#donutChart circle[data-dash]");
  circles.forEach((c, i) => {
    setTimeout(() => {
      c.style.strokeDasharray = c.dataset.dash;
    }, 140 * i);
  });

  document.querySelectorAll(".bar-fill").forEach((bar, i) => {
    setTimeout(() => {
      bar.style.height = bar.dataset.h + "%";
    }, 70 * i);
  });
}

/* ============================================================
   初始化
   ============================================================ */
function init() {
  renderKpis();
  renderDonut();
  renderBars();
  renderAgents();

  const year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();

  /* 滚动入场 + 数字动画 */
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in-view");
        entry.target.querySelectorAll("[data-value]").forEach((el) => {
          if (el.dataset.done) return;
          el.dataset.done = "1";
          countUp(el);
        });
        revealObserver.unobserve(entry.target);
      });
    },
    { threshold: 0.16, rootMargin: "0px 0px -40px 0px" }
  );

  document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

  /* 报告图表：进入视野时播放 */
  const report = document.getElementById("report");
  let chartsPlayed = false;
  const chartObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting || chartsPlayed) return;
        chartsPlayed = true;
        animateCharts();
      });
    },
    { threshold: 0.15 }
  );
  chartObserver.observe(report);

  /* 导航高亮 */
  const navLinks = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  const spyObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const href = "#" + entry.target.id;
        navLinks.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === href);
        });
      });
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  document.querySelectorAll("main section[id]").forEach((s) => spyObserver.observe(s));

  /* 移动端菜单 */
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");

  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  links.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  });
}

init();
