/* Skin Retouch console — no framework, no build step.
   Everything the page needs comes from two endpoints: /api/health (environment
   and limits) and /api/schema (parameter definitions). The DOM is generated from
   the schema, so a new knob in server/config.py PARAM_SPECS shows up here
   without touching this file. */

(() => {
  "use strict";

  const POLL_MS = 350;
  const $ = (id) => document.getElementById(id);

  const el = {
    status: $("status"),
    statusText: $("status-text"),
    chipDevice: $("chip-device"),
    chipVersion: $("chip-version"),
    btnLogin: $("btn-login"),
    presets: $("presets"),
    params: $("params"),
    btnReset: $("btn-reset"),
    btnRun: $("btn-run"),
    runHint: $("run-hint"),
    stage: $("stage"),
    tabs: Array.from(document.querySelectorAll(".tab")),
    btnZoom: $("btn-zoom"),
    btnSwap: $("btn-swap"),
    btnDownload: $("btn-download"),
    canvas: $("canvas"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    dzLimit: $("dz-limit"),
    compare: $("compare"),
    imgOut: $("img-out"),
    imgSrc: $("img-src"),
    divider: $("divider"),
    single: $("single"),
    imgSingle: $("img-single"),
    diag: $("diag"),
    busy: $("busy"),
    phases: Array.from(document.querySelectorAll("#phases span")),
    busyTimer: $("busy-timer"),
    busyNote: $("busy-note"),
    btnCancel: $("btn-cancel"),
    busyBar: $("busy-bar"),
    metaSize: $("meta-size"),
    metaTime: $("meta-time"),
    metaCover: $("meta-cover"),
    metaNote: $("meta-note"),
    footEnv: $("foot-env"),
    toasts: $("toasts"),
    gate: $("gate"),
    gateForm: $("gate-form"),
    gateInput: $("gate-input"),
    gateError: $("gate-error"),
  };

  const state = {
    schema: null,
    health: null,
    params: {},
    file: null,
    localUrl: "",
    jobId: "",
    job: null,
    busy: false,
    pollTimer: 0,
    clockTimer: 0,
    startedAt: 0,
    tab: "compare",
    swapped: false,
    split: 50,
    urls: [],
  };

  /* ------------------------------------------------------------- helpers */

  function setStatus(kind, text) {
    el.status.dataset.state = kind;
    el.statusText.textContent = text;
  }

  function toast(message, kind = "info", ttl = 4200) {
    const node = document.createElement("div");
    node.className = "toast";
    node.dataset.kind = kind;
    node.textContent = message;
    el.toasts.appendChild(node);
    setTimeout(() => {
      node.classList.add("leaving");
      setTimeout(() => node.remove(), 260);
    }, ttl);
  }

  function fmtBytes(n) {
    if (!n && n !== 0) return "—";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
    return (n / 1024 / 1024).toFixed(1) + " MB";
  }

  function fmtMs(ms) {
    if (!ms && ms !== 0) return "—";
    return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
  }

  function num(value, digits = 2) {
    return Number(value).toFixed(digits);
  }

  function trackUrl(url) {
    if (url && !state.urls.includes(url)) state.urls.push(url);
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function api(path, options = {}) {
    const init = Object.assign({ credentials: "same-origin" }, options);
    const res = await fetch(path, init);
    let body = null;
    const ctype = res.headers.get("Content-Type") || "";
    if (ctype.includes("application/json")) {
      try {
        body = await res.json();
      } catch {
        body = null;
      }
    }
    if (res.status === 401) {
      requestToken("需要令牌才能继续。");
      throw new Error((body && body.message) || "未授权");
    }
    if (!res.ok) {
      const msg = (body && (body.message || body.error)) || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return body;
  }

  /* ------------------------------------------------------------ token UI */

  function requestToken(message) {
    if (!el.gate.hidden) return;
    el.gateError.textContent = message || "";
    el.gate.hidden = false;
    setTimeout(() => el.gateInput.focus(), 60);
  }

  async function login(token) {
    const res = await fetch("/api/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (!res.ok) {
      let msg = "令牌不正确。";
      try {
        const body = await res.json();
        if (body && body.message) msg = body.message;
      } catch {
        /* keep the default message */
      }
      throw new Error(msg);
    }
    return true;
  }

  el.gateForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const token = el.gateInput.value.trim();
    if (!token) return;
    el.gateError.textContent = "";
    try {
      await login(token);
      el.gate.hidden = true;
      el.gateInput.value = "";
      await boot({ quiet: true });
      toast("已解锁", "ok");
    } catch (err) {
      el.gateError.textContent = err.message;
    }
  });

  el.btnLogin.addEventListener("click", () => requestToken(""));

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !el.gate.hidden) {
      el.gate.hidden = true;
    }
  });

  /* --------------------------------------------------------------- boot */

  async function boot({ quiet = false } = {}) {
    let health;
    try {
      health = await api("/api/health");
    } catch (err) {
      if (err.message !== "未授权") {
        setStatus("error", "服务未响应");
        if (!quiet) toast("连不上服务：" + err.message, "error", 8000);
      }
      return false;
    }
    state.health = health;
    state.repaint = health.repaint || null;
    el.chipVersion.textContent = "v" + (health.version || "—");

    if (health.auth_required && !health.authorized) {
      setStatus("locked", "已加锁");
      requestToken("这个服务要求令牌。");
      return false;
    }
    el.btnLogin.hidden = !health.auth_required;

    // /api/health gives the device flags only once authorized; the unpaid
    // branch above still tells us whether the token is plausible.
    const caps = health.caps || {};
    const gpuName = caps.gpu || "CUDA";
    el.chipDevice.textContent = caps.cuda
      ? "GPU · " + (gpuName.length > 30 ? gpuName.slice(0, 29) + "…" : gpuName)
      : "CPU";
    el.chipDevice.title = caps.cuda
      ? `CUDA ${caps.torch || ""} · OpenCV ${caps.opencv || ""}`
      : `仅 CPU · OpenCV ${caps.opencv || ""}`;
    el.footEnv.textContent = [
      caps.torch ? "torch " + String(caps.torch).split("+")[0] : "",
      caps.opencv ? "cv2 " + caps.opencv : "",
      caps.models && caps.models.face ? "face✓" : "",
      caps.models && caps.models.person ? "person✓" : "",
      health.repaint && health.repaint.available ? "inpaint✓" : "",
    ]
      .filter(Boolean)
      .join(" · ");

    setStatus("online", "就绪");

    if (!state.schema) {
      state.schema = await api("/api/schema");
      buildPresets();
      buildParams();
      el.dzLimit.textContent = fmtBytes(state.schema.max_upload_bytes);
      const mp = health.limits ? health.limits.megapixels : null;
      if (mp) el.metaNote.textContent = `上限 ${mp} MP · 首跑约 20 秒加载检测器`;
    }
    noteRepaint();
    return true;
  }

  // The repaint switch must be honest about the engine behind it. When ComfyUI
  // is not running there is nothing to redraw with, and a ticked box that
  // silently does nothing is the worst possible outcome -- so the row says so
  // before the job is submitted, and the job's own warning says it again.
  function noteRepaint() {
    const row = el.params.querySelector('[data-key="repaint"]');
    if (!row) return;
    const engine = state.repaint || {};
    const available = !!engine.available;
    row.dataset.unavailable = available ? "0" : "1";
    const hint = row.querySelector(".param-hint");
    if (hint) {
      if (available) {
        hint.textContent = "把痘点真正重画一遍而不是抹平，每块约 10 秒。与磨皮叠加效果最好。";
      } else if (engine.state === "starting") {
        hint.textContent = "引擎正在启动，模型加载完就能勾选。";
      } else {
        hint.textContent = "引擎未就绪："
          + (engine.spawnable === false
            ? (engine.detail || "找不到 ComfyUI，用 --comfyui-dir 指定")
            : (engine.url || "ComfyUI") + " 上没有响应，可点上方「启动」。");
      }
    }
    for (const key of ["repaint_denoise", "repaint_steps"]) {
      const other = el.params.querySelector(`[data-key="${key}"]`);
      if (other) other.dataset.unavailable = available ? "0" : "1";
    }
    renderEngine();
  }

  /* --------------------------------------------------- repaint engine chip */

  const engineUI = { node: null, btn: null };

  function engineControl() {
    const wrap = document.createElement("span");
    wrap.className = "engine";
    const dot = document.createElement("i");
    dot.className = "engine-dot";
    dot.setAttribute("aria-hidden", "true");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "engine-btn";
    btn.addEventListener("click", toggleEngine);
    wrap.append(dot, btn);
    engineUI.node = wrap;
    engineUI.btn = btn;
    return wrap;
  }

  // The chip reads the same status the health poll uses, so a ComfyUI started
  // outside this page also shows up here without a reload.
  function renderEngine() {
    const wrap = engineUI.node;
    if (!wrap) return;
    const engine = state.repaint || {};
    const key = engine.available ? "ready" : (engine.state || "idle");
    wrap.dataset.state = key;
    wrap.title = engine.detail || engine.url || "";
    const btn = engineUI.btn;
    if (key === "starting") {
      btn.textContent = "启动中…";
      btn.disabled = true;
    } else if (key === "ready") {
      btn.textContent = engine.managed ? "停止" : "占用中";
      btn.disabled = !engine.managed;  // never stop someone else's ComfyUI
    } else if (engine.spawnable === false) {
      btn.textContent = "不可用";
      btn.disabled = true;
    } else {
      btn.textContent = key === "failed" ? "重试" : "启动";
      btn.disabled = false;
    }
  }

  async function toggleEngine() {
    const engine = state.repaint || {};
    const action = engine.available && engine.managed ? "stop" : "start";
    engineUI.btn.disabled = true;
    try {
      state.repaint = await api("/api/repaint/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      renderEngine();
      noteRepaint();
      if (action === "start" && state.repaint && state.repaint.state === "starting") {
        toast("正在启动重绘引擎，首次加载模型约 1 分钟。");
        pollEngine();
      } else if (action === "stop") {
        toast("重绘引擎已停止。");
      }
    } catch (err) {
      toast("引擎操作失败：" + err.message);
    } finally {
      renderEngine();
    }
  }

  // Loading a 6.9 GB checkpoint takes a while; keep asking so the chip turns
  // green on its own instead of waiting for the next slow health poll.
  async function pollEngine() {
    for (let i = 0; i < 150; i += 1) {
      await sleep(2000);
      try {
        state.repaint = await api("/api/repaint/engine");
      } catch {
        return;
      }
      renderEngine();
      noteRepaint();
      if (!state.repaint || state.repaint.state !== "starting") {
        if (state.repaint && state.repaint.available) toast("重绘引擎已就绪。");
        return;
      }
    }
  }

  /* ---------------------------------------------------------- parameter UI */

  function buildPresets() {
    el.presets.textContent = "";
    for (const preset of state.schema.presets || []) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset";
      btn.textContent = preset.label;
      btn.dataset.preset = preset.key;
      btn.setAttribute("aria-pressed", "false");
      btn.addEventListener("click", () => applyPreset(preset));
      el.presets.appendChild(btn);
    }
  }

  function markPreset(key) {
    for (const btn of el.presets.querySelectorAll(".preset")) {
      btn.setAttribute("aria-pressed", String(btn.dataset.preset === key));
    }
  }

  function applyPreset(preset) {
    Object.assign(state.params, preset.params);
    markPreset(preset.key);
    syncParamUI();
  }

  function defaultValue(spec) {
    if (spec.key in state.params) return state.params[spec.key];
    return spec.default;
  }

  function buildParams() {
    el.params.textContent = "";
    state.params = Object.assign({}, state.schema.defaults || {});
    let lastGroup = null;

    for (const spec of state.schema.params) {
      if (spec.group !== lastGroup) {
        lastGroup = spec.group;
        const title = document.createElement("div");
        title.className = "group-title";
        const label = document.createElement("span");
        label.textContent = spec.group;
        title.appendChild(label);
        // The repaint group owns a GPU engine that has to be started and can be
        // stopped again; its state belongs on the heading, not in a paragraph.
        if (spec.group === "重绘") title.appendChild(engineControl());
        el.params.appendChild(title);
      }
      el.params.appendChild(rowFor(spec));
    }
    syncParamUI();
    syncParamsOverflow();
  }

  // The parameter list is taller than the panel on most screens; feather the
  // bottom edge so it reads as "scrolls" rather than "cut off".
  function syncParamsOverflow() {
    const box = el.params;
    if (!box) return;
    const overflow = box.scrollHeight > box.clientHeight + 2;
    box.dataset.overflow = overflow ? "1" : "0";
    const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    box.dataset.atend = atEnd ? "1" : "0";
  }

  function rowFor(spec) {
    if (spec.kind === "bool") return boolRow(spec);
    if (spec.kind === "enum") return enumRow(spec);
    return numericRow(spec);
  }

  function numericRow(spec) {
    const wrap = document.createElement("div");
    wrap.className = "param";
    wrap.dataset.key = spec.key;

    const top = document.createElement("div");
    top.className = "param-top";
    const name = document.createElement("span");
    name.className = "param-name";
    name.textContent = spec.label;
    const key = document.createElement("span");
    key.className = "param-key";
    key.textContent = spec.key;
    const value = document.createElement("span");
    value.className = "param-value";
    top.append(name, key, value);

    const input = document.createElement("input");
    input.type = "range";
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.dataset.key = spec.key;
    input.setAttribute("aria-label", spec.label);
    input.addEventListener("input", () => {
      const raw = Number(input.value);
      state.params[spec.key] = spec.kind === "int" ? Math.round(raw) : raw;
      markPreset("");
      paintRow(spec, wrap);
    });

    wrap.append(top, input);
    if (spec.hint) {
      const hint = document.createElement("div");
      hint.className = "param-hint";
      hint.textContent = spec.hint;
      wrap.appendChild(hint);
    }
    return wrap;
  }

  function boolRow(spec) {
    const wrap = document.createElement("div");
    wrap.className = "param";
    wrap.dataset.key = spec.key;

    const top = document.createElement("div");
    top.className = "param-top";
    const name = document.createElement("span");
    name.className = "param-name";
    name.textContent = spec.label;
    const key = document.createElement("span");
    key.className = "param-key";
    key.textContent = spec.key;
    top.append(name, key);

    const row = document.createElement("div");
    row.className = "switch-row";
    const sw = document.createElement("button");
    sw.type = "button";
    sw.className = "switch";
    sw.setAttribute("role", "switch");
    sw.setAttribute("aria-label", spec.label);
    sw.addEventListener("click", () => {
      state.params[spec.key] = !state.params[spec.key];
      markPreset("");
      paintRow(spec, wrap);
    });
    row.append(sw);

    wrap.append(top, row);
    if (spec.hint) {
      const hint = document.createElement("div");
      hint.className = "param-hint";
      hint.textContent = spec.hint;
      wrap.appendChild(hint);
    }
    return wrap;
  }

  function enumRow(spec) {
    const wrap = document.createElement("div");
    wrap.className = "param";
    wrap.dataset.key = spec.key;

    const top = document.createElement("div");
    top.className = "param-top";
    const name = document.createElement("span");
    name.className = "param-name";
    name.textContent = spec.label;
    const key = document.createElement("span");
    key.className = "param-key";
    key.textContent = spec.key;
    top.append(name, key);

    const seg = document.createElement("div");
    seg.className = "segmented";
    seg.setAttribute("role", "group");
    for (const option of spec.options) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = option;
      btn.dataset.option = option;
      btn.setAttribute("aria-pressed", "false");
      btn.addEventListener("click", () => {
        state.params[spec.key] = option;
        markPreset("");
        paintRow(spec, wrap);
      });
      seg.appendChild(btn);
    }

    wrap.append(top, seg);
    if (spec.hint) {
      const hint = document.createElement("div");
      hint.className = "param-hint";
      hint.textContent = spec.hint;
      wrap.appendChild(hint);
    }
    return wrap;
  }

  function specOf(key) {
    return state.schema.params.find((s) => s.key === key);
  }

  function paintRow(spec, wrap) {
    const value = state.params[spec.key];
    if (spec.kind === "bool") {
      const sw = wrap.querySelector(".switch");
      if (sw) sw.setAttribute("aria-checked", String(!!value));
      return;
    }
    if (spec.kind === "enum") {
      for (const btn of wrap.querySelectorAll(".segmented button")) {
        btn.setAttribute("aria-pressed", String(btn.dataset.option === value));
      }
      return;
    }
    const input = wrap.querySelector('input[type="range"]');
    const label = wrap.querySelector(".param-value");
    if (input) {
      input.value = String(value);
      const span = spec.max - spec.min;
      const fill = span > 0 ? ((Number(value) - spec.min) / span) * 100 : 0;
      input.style.setProperty("--fill", Math.max(0, Math.min(100, fill)) + "%");
    }
    if (label) {
      label.textContent =
        spec.kind === "int" ? String(value) : num(value, 2);
    }
  }

  function syncParamUI() {
    for (const wrap of el.params.querySelectorAll(".param")) {
      const spec = specOf(wrap.dataset.key);
      if (spec) paintRow(spec, wrap);
    }
  }

  el.btnReset.addEventListener("click", () => {
    state.params = Object.assign({}, state.schema.defaults || {});
    markPreset("");
    syncParamUI();
  });

  /* -------------------------------------------------------- file handling */

  function acceptFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type || "") && !/\.(png|jpe?g|tiff?|webp|bmp)$/i.test(file.name)) {
      toast("这不是图片文件。", "error");
      return;
    }
    const limit = (state.schema && state.schema.max_upload_bytes) || 128 * 1024 * 1024;
    if (file.size > limit) {
      toast(`文件 ${fmtBytes(file.size)} 超过上限 ${fmtBytes(limit)}。`, "error", 6000);
      return;
    }
    state.file = file;
    if (state.localUrl) URL.revokeObjectURL(state.localUrl);
    state.localUrl = URL.createObjectURL(file);
    el.dropzone.hidden = true;
    el.btnRun.disabled = state.busy;
    el.runHint.textContent = "按 Ctrl / ⌘ + ↵ 开始";
    el.metaSize.textContent = fmtBytes(file.size);
    el.metaTime.textContent = "—";
    el.metaCover.textContent = "—";
    el.metaNote.textContent = file.name.length > 42 ? file.name.slice(0, 40) + "…" : file.name;
    setTab(state.job ? "compare" : "result");
    if (!state.job) showLocalPreview();
  }

  function showLocalPreview() {
    if (!state.localUrl) return;
    trackUrl(state.localUrl);
    el.imgSingle.src = state.localUrl;
    el.compare.hidden = true;
    el.single.hidden = state.tab !== "result";
  }

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    acceptFile(el.fileInput.files && el.fileInput.files[0]);
    el.fileInput.value = "";
  });

  for (const event of ["dragenter", "dragover"]) {
    el.stage.addEventListener(event, (ev) => {
      if (!ev.dataTransfer || !Array.from(ev.dataTransfer.types || []).includes("Files")) return;
      ev.preventDefault();
      if (!el.dropzone.hidden) el.dropzone.classList.add("dragging");
    });
  }
  el.stage.addEventListener("dragleave", (ev) => {
    if (ev.target === el.stage) el.dropzone.classList.remove("dragging");
  });
  el.stage.addEventListener("drop", (ev) => {
    if (!ev.dataTransfer) return;
    ev.preventDefault();
    el.dropzone.classList.remove("dragging");
    acceptFile(ev.dataTransfer.files && ev.dataTransfer.files[0]);
  });

  /* ---------------------------------------------------------------- submit */

  function queryFor(params, name) {
    const qs = new URLSearchParams();
    for (const spec of state.schema.params) {
      const value = params[spec.key];
      if (value === undefined || value === null) continue;
      qs.set(spec.key, spec.kind === "bool" ? (value ? "1" : "0") : String(value));
    }
    if (name) qs.set("name", name);
    return qs.toString();
  }

  async function run() {
    if (state.busy || !state.file || !state.schema) return;
    state.busy = true;
    state.job = null;
    state.jobId = "";
    el.btnRun.disabled = true;
    el.btnSwap.hidden = true;
    el.btnDownload.hidden = true;
    el.runHint.textContent = "处理中…";
    setStatus("busy", "处理中");
    startBusy();
    let submitted;
    try {
      submitted = await api("/api/jobs?" + queryFor(state.params, state.file.name), {
        method: "POST",
        headers: { "Content-Type": state.file.type || "application/octet-stream" },
        body: state.file,
      });
    } catch (err) {
      state.busy = false;
      stopBusy();
      el.btnRun.disabled = false;
      el.runHint.textContent = "按 Ctrl / ⌘ + ↵ 重试";
      setStatus("error", "提交失败");
      toast("提交失败：" + err.message, "error", 8000);
      return;
    }
    state.jobId = submitted.id;
    trackJob(submitted);
    poll();
    if (state.params.repaint && !(state.repaint && state.repaint.available)) {
      toast("局部重绘引擎未就绪，本次只做磨皮。", "info", 6000);
    }
  }

  function trackJob(snapshot) {
    state.job = snapshot;
    paintBusy(snapshot);
  }

  async function poll() {
    const id = state.jobId;
    if (!id) return;
    let snapshot;
    try {
      snapshot = await api("/api/jobs/" + id);
    } catch (err) {
      if (err.message === "未授权") return;
      await sleep(POLL_MS * 2);
      if (state.jobId === id) poll();
      return;
    }
    if (state.jobId !== id) return;
    trackJob(snapshot);

    if (snapshot.status === "done") {
      finish(snapshot);
      return;
    }
    if (snapshot.status === "error" || snapshot.status === "canceled") {
      fail(snapshot);
      return;
    }
    state.pollTimer = setTimeout(poll, POLL_MS);
  }

  function finish(snapshot) {
    state.busy = false;
    stopBusy();
    el.btnRun.disabled = false;
    el.runHint.textContent = "改参数再跑一次";
    setStatus("online", "就绪");
    lsSet("skin.lastJob", snapshot.id);
    paintResult(snapshot);
    if (snapshot.cold_start) {
      toast("首次运行已加载检测器，后续会快很多。", "ok", 6000);
    }
    for (const warning of snapshot.warnings || []) {
      toast("注意：" + warning, "info", 7000);
    }
  }

  function fail(snapshot) {
    state.busy = false;
    stopBusy();
    el.btnRun.disabled = false;
    el.runHint.textContent = "调整后再试一次";
    setStatus("error", "失败");
    const message = snapshot.error || "处理失败，没有详细原因。";
    el.diag.textContent = "";
    const box = document.createElement("div");
    box.className = "stat";
    const dt = document.createElement("dt");
    dt.textContent = "错误";
    const dd = document.createElement("dd");
    dd.style.fontSize = "12px";
    dd.style.color = "var(--danger)";
    dd.textContent = message;
    box.append(dt, dd);
    el.diag.appendChild(box);
    setTab("diag");
    toast(message, "error", 9000);
  }

  /* ---------------------------------------------------------------- restore */

  // A refresh used to throw the result away. Keep the last finished job
  // addressable so `?job=<id>` (shareable) and a plain reload both reopen it.
  async function restore(jobId) {
    if (!/^[0-9a-f]{16}$/.test(jobId)) return false;
    let snapshot;
    try {
      snapshot = await api("/api/jobs/" + jobId);
    } catch {
      return false;
    }
    if (snapshot.status === "queued" || snapshot.status === "running") {
      state.jobId = snapshot.id;
      state.busy = true;
      el.btnRun.disabled = true;
      setStatus("busy", "处理中");
      startBusy();
      trackJob(snapshot);
      poll();
      return true;
    }
    if (snapshot.status !== "done") return false;

    state.jobId = snapshot.id;
    state.job = snapshot;
    state.params = Object.assign({}, state.params, snapshot.params || {});
    syncParamUI();
    paintResult(snapshot);
    el.btnRun.disabled = false;
    el.runHint.textContent = "改参数再跑一次";
    setStatus("online", "就绪");

    // Pull the untouched original so the restored view can still be re-run.
    try {
      const res = await fetch(`/api/jobs/${snapshot.id}/result?kind=source`, {
        credentials: "same-origin",
      });
      if (res.ok) {
        const blob = await res.blob();
        const name = (snapshot.source && snapshot.source.name) || "source.png";
        state.file = new File([blob], name, { type: blob.type || "image/png" });
      }
    } catch {
      // Previews still render without the original; only re-running needs it.
    }
    return true;
  }

  function lsGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function lsSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      // Private mode / storage disabled: restoring is a nicety, not a feature.
    }
  }

  /* ------------------------------------------------------------- busy view */

  function startBusy() {
    el.busy.hidden = false;
    state.startedAt = performance.now();
    for (const span of el.phases) {
      delete span.dataset.active;
      delete span.dataset.done;
    }
    el.busyBar.style.width = "0%";
    el.busyTimer.textContent = "0.0 s";
    el.busyNote.textContent = (state.health && state.health.queue && state.health.queue.pending)
      ? "排队 " + state.health.queue.pending
      : "";
    state.clockTimer = setInterval(() => {
      el.busyTimer.textContent = ((performance.now() - state.startedAt) / 1000).toFixed(1) + " s";
    }, 100);
  }

  function stopBusy() {
    el.busy.hidden = true;
    clearInterval(state.clockTimer);
    state.clockTimer = 0;
  }

  // The repaint phase only exists when the user asked for it, so the step list
  // and the progress mapping are both derived from the job's own params.
  const PHASE_ORDER = ["decode", "pipeline", "repaint", "encode"];
  const PHASE_ORDER_PLAIN = ["decode", "pipeline", "encode"];
  const PHASE_WIDTHS = {
    on: { queue: 0.06, decode: 0.14, pipeline: 0.5, repaint: 0.88, encode: 0.96, done: 1 },
    off: { queue: 0.08, decode: 0.2, pipeline: 0.72, encode: 0.94, done: 1 },
  };

  function paintBusy(snapshot) {
    if (el.busy.hidden) return;
    const repaintOn = !!(snapshot.params && snapshot.params.repaint);
    const order = repaintOn ? PHASE_ORDER : PHASE_ORDER_PLAIN;
    const phase = snapshot.phase || "queued";
    const index = order.indexOf(phase);
    for (const span of el.phases) {
      const name = span.dataset.phase;
      span.hidden = name === "repaint" && !repaintOn;
      const position = order.indexOf(name);
      if (index < 0) {
        delete span.dataset.active;
        delete span.dataset.done;
        if (name === "decode") span.dataset.active = "true";
      } else if (position < index) {
        span.dataset.done = "true";
        delete span.dataset.active;
      } else if (position === index) {
        span.dataset.active = "true";
        delete span.dataset.done;
      } else {
        delete span.dataset.active;
        delete span.dataset.done;
      }
    }
    const widths = repaintOn ? PHASE_WIDTHS.on : PHASE_WIDTHS.off;
    const width = widths[phase] || 0.12;
    el.busyBar.style.width = Math.round(width * 100) + "%";
    const note = [];
    if (snapshot.queue_position > 0) note.push("队列第 " + snapshot.queue_position + " 位");
    else if (snapshot.status === "running") {
      note.push(phase === "repaint" ? "局部重绘中，每块约 10 秒" : "单张串行处理");
    }
    el.busyNote.textContent = note.join(" · ");
    // Only a still-queued job can actually be pulled; the pipeline has no
    // interruption points once it starts, so don't offer a dead button.
    el.btnCancel.hidden = snapshot.status !== "queued";
  }

  el.btnCancel.addEventListener("click", async () => {
    if (!state.jobId) return;
    el.btnCancel.disabled = true;
    try {
      await api("/api/jobs/" + state.jobId, { method: "DELETE" });
      state.jobId = "";
      state.busy = false;
      stopBusy();
      el.btnRun.disabled = false;
      el.runHint.textContent = "已取消，可以重新开始";
      setStatus("online", "就绪");
      toast("已从队列移除。", "ok");
    } catch (err) {
      toast("取消失败：" + err.message, "error");
    } finally {
      el.btnCancel.disabled = false;
      el.btnCancel.hidden = true;
    }
  });

  /* --------------------------------------------------------------- results */

  function paintResult(snapshot) {
    const id = snapshot.id;
    const stamp = "v=" + encodeURIComponent(String(snapshot.finished_at || Date.now()));
    const src = `/api/jobs/${id}/result?kind=preview_source&${stamp}`;
    const out = `/api/jobs/${id}/result?kind=preview_out&${stamp}`;
    const mask = `/api/jobs/${id}/result?kind=preview_mask&${stamp}`;
    trackUrl(src);
    trackUrl(out);
    trackUrl(mask);

    state.swapped = false;
    el.imgSrc.src = src;
    el.imgOut.src = out;
    el.imgSingle.src = out;
    el.compare.hidden = false;
    el.dropzone.hidden = true;
    el.btnSwap.hidden = false;
    el.btnSwap.textContent = "交换";

    const image = snapshot.image || {};
    el.metaSize.textContent = image.width
      ? `${image.width} × ${image.height} · ${image.megapixels} MP`
      : fmtBytes(snapshot.source && snapshot.source.bytes);
    el.metaTime.textContent = fmtMs(snapshot.elapsed_ms);
    const coverage = coverageOf(snapshot.info);
    el.metaCover.textContent = coverage === null ? "无皮肤" : "覆盖 " + (coverage * 100).toFixed(1) + "%";
    el.metaNote.textContent = snapshot.source && snapshot.source.name
      ? snapshot.source.name
      : "完成";

    el.btnDownload.hidden = false;
    el.btnDownload.href = `/api/jobs/${id}/result?kind=out&download=1`;
    el.btnDownload.setAttribute("download", downloadName(snapshot));

    setSplit(50);
    setTab("compare");
    renderDiag(snapshot);
    setZoom("fit");
    if (!coverage) toast("没检测到皮肤区域，检查阈值或换个角度。", "info", 6000);
  }

  function downloadName(snapshot) {
    const name = (snapshot.source && snapshot.source.name) || "image";
    const base = name.replace(/\.[^.]+$/, "").replace(/[\\/:*?"<>|]/g, "_") || "image";
    return base + "-retouched.png";
  }

  function coverageOf(info) {
    if (!Array.isArray(info) || !info.length) return null;
    let total = 0;
    for (const item of info) {
      const value = Number(item && item.coverage);
      if (Number.isFinite(value)) total += value;
    }
    return total > 0 ? Math.min(total, 1) : null;
  }

  function renderDiag(snapshot) {
    el.diag.textContent = "";
    const image = snapshot.image || {};
    const timing = snapshot.timing || {};
    const grid = document.createElement("div");
    grid.className = "diag-grid";
    const stats = [
      ["分辨率", image.width ? `${image.width} × ${image.height}` : "—"],
      ["总耗时", fmtMs(snapshot.elapsed_ms)],
      ["美肤计算", fmtMs(timing.pipeline_ms)],
      ["皮肤覆盖", coverageOf(snapshot.info) === null
        ? "—"
        : (coverageOf(snapshot.info) * 100).toFixed(1) + " %"],
    ];
    const repaint = snapshot.repaint || {};
    if (Number.isFinite(Number(timing.repaint_ms))) {
      stats.push(["局部重绘", fmtMs(timing.repaint_ms)]);
      stats.push(["重绘区域", repaint.tiles_done
        ? `${repaint.tiles_done} 块 · ${((Number(repaint.mask_frac) || 0) * 100).toFixed(2)} %`
        : (repaint.reason || "无瑕疵超过阈值")]);
    } else if (repaint.reason) {
      stats.push(["局部重绘", repaint.reason]);
    }
    for (const [label, value] of stats) {
      const box = document.createElement("div");
      box.className = "stat";
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      box.append(dt, dd);
      grid.appendChild(box);
    }
    el.diag.appendChild(grid);

    if (Array.isArray(snapshot.info) && snapshot.info.length) {
      const table = document.createElement("table");
      table.className = "info";
      const head = document.createElement("tr");
      for (const label of ["对象", "位置", "尺寸", "覆盖", "备注"]) {
        const th = document.createElement("th");
        th.textContent = label;
        head.appendChild(th);
      }
      table.appendChild(head);
      let faceIndex = 0;
      snapshot.info.forEach((item) => {
        const row = document.createElement("tr");
        const isBody = item.kind === "body";
        if (!isBody) faceIndex += 1;
        const kind = isBody ? "身体" : "人脸 " + faceIndex;
        const box = Array.isArray(item.box) ? item.box : null;
        const size = item.face_px
          ? Math.round(item.face_px) + " px"
          : item.scale_px
          ? Math.round(item.scale_px) + " px"
          : "—";
        const note = [];
        if (item.score !== undefined) note.push("置信 " + Number(item.score).toFixed(2));
        if (item.ref) note.push("参考 " + item.ref);
        if (item.backend) note.push(item.backend);
        if (Number.isFinite(Number(item.strips))) note.push(Number(item.strips) + " 条带");
        if (item.feather_px) note.push("羽化 " + Math.round(item.feather_px) + " px");
        if (item.roi) note.push(`${item.roi[0]}×${item.roi[1]}`);
        const cells = [
          kind,
          box ? `x${box[0]} y${box[1]} w${box[2]} h${box[3]}` : "—",
          size,
          Number.isFinite(Number(item.coverage)) ? (Number(item.coverage) * 100).toFixed(2) + " %" : "—",
          note.join(" · ") || "—",
        ];
        for (const value of cells) {
          const td = document.createElement("td");
          td.textContent = value;
          row.appendChild(td);
        }
        table.appendChild(row);
      });
      el.diag.appendChild(table);
    }

    const segments = ["decode", "pipeline", "repaint", "encode"]
      .map((key) => [key, timing[key + "_ms"]])
      .filter((pair) => Number.isFinite(Number(pair[1])));
    if (segments.length) {
      const wrap = document.createElement("div");
      wrap.className = "timing";
      const peak = Math.max(...segments.map((pair) => Number(pair[1])), 1);
      const names = { decode: "解码", pipeline: "美肤计算", repaint: "局部重绘", encode: "编码" };
      for (const [key, ms] of segments) {
        const row = document.createElement("div");
        row.className = "timing-row";
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = names[key] || key;
        const track = document.createElement("span");
        track.className = "track";
        const bar = document.createElement("i");
        bar.style.width = Math.round((Number(ms) / peak) * 100) + "%";
        track.appendChild(bar);
        const value = document.createElement("span");
        value.className = "ms";
        value.textContent = fmtMs(Number(ms));
        row.append(name, track, value);
        wrap.appendChild(row);
      }
      el.diag.appendChild(wrap);
    }
  }

  /* ----------------------------------------------------------------- views */

  function setTab(name) {
    state.tab = name;
    for (const tab of el.tabs) {
      tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
    }
    const hasResult = !!state.job && state.job.status === "done";
    el.compare.hidden = !(name === "compare" && hasResult);
    el.single.hidden = name !== "result" && name !== "mask";
    el.diag.hidden = name !== "diag";
    if (name === "result") {
      el.imgSingle.src = hasResult ? state.swapped ? sourceUrl() : outUrl() : state.localUrl;
    }
    if (name === "mask") {
      el.imgSingle.src = hasResult ? maskUrl() : state.localUrl;
    }
    if (hasResult) {
      const note = {
        compare: "拖动分割线对比修前修后",
        result: state.swapped ? "当前显示原图" : "当前显示修后成图",
        mask: "白色 = 会被美肤的区域，五官保留为黑",
        diag: (state.job && state.job.source && state.job.source.name) || "诊断",
      }[name];
      if (note) el.metaNote.textContent = note;
    }
  }

  function outUrl() {
    return state.jobId ? `/api/jobs/${state.jobId}/result?kind=preview_out` : "";
  }

  function sourceUrl() {
    return state.jobId ? `/api/jobs/${state.jobId}/result?kind=preview_source` : "";
  }

  function maskUrl() {
    return state.jobId ? `/api/jobs/${state.jobId}/result?kind=preview_mask` : "";
  }

  for (const tab of el.tabs) {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
  }

  function setZoom(mode) {
    el.stage.dataset.zoom = mode;
    el.btnZoom.textContent = mode === "actual" ? "1:1" : "适应";
  }

  el.btnZoom.addEventListener("click", () => {
    setZoom(el.stage.dataset.zoom === "actual" ? "fit" : "actual");
  });

  el.btnSwap.addEventListener("click", () => {
    if (!state.job || state.job.status !== "done") return;
    state.swapped = !state.swapped;
    const out = outUrl();
    const src = sourceUrl();
    el.imgOut.src = state.swapped ? src : out;
    el.imgSrc.src = state.swapped ? out : src;
    const left = el.compare.querySelector(".tag-left");
    const right = el.compare.querySelector(".tag-right");
    if (left && right) {
      const text = state.swapped ? ["修后", "原图"] : ["原图", "修后"];
      left.textContent = text[0];
      right.textContent = text[1];
    }
    setSplit(100 - state.split);
  });

  /* --------------------------------------------------------------- slider */

  function setSplit(percent) {
    const clamped = Math.max(0, Math.min(100, percent));
    state.split = clamped;
    el.compare.style.setProperty("--split", clamped + "%");
    el.divider.setAttribute("aria-valuenow", String(Math.round(clamped)));
  }

  (function bindSlider() {
    let dragging = false;
    const move = (ev) => {
      const rect = el.compare.getBoundingClientRect();
      if (!rect.width) return;
      setSplit(((ev.clientX - rect.left) / rect.width) * 100);
    };
    el.compare.addEventListener("pointerdown", (ev) => {
      if (ev.button !== undefined && ev.button !== 0) return;
      dragging = true;
      el.compare.setPointerCapture(ev.pointerId);
      move(ev);
      ev.preventDefault();
    });
    el.compare.addEventListener("pointermove", (ev) => {
      if (dragging) move(ev);
    });
    const release = (ev) => {
      dragging = false;
      try {
        el.compare.releasePointerCapture(ev.pointerId);
      } catch {
        /* pointer already released */
      }
    };
    el.compare.addEventListener("pointerup", release);
    el.compare.addEventListener("pointercancel", () => {
      dragging = false;
    });

    el.divider.tabIndex = 0;
    el.divider.setAttribute("role", "slider");
    el.divider.setAttribute("aria-label", "对比分割线");
    el.divider.setAttribute("aria-valuemin", "0");
    el.divider.setAttribute("aria-valuemax", "100");
    el.divider.addEventListener("keydown", (ev) => {
      const step = ev.shiftKey ? 10 : 2;
      if (ev.key === "ArrowLeft") {
        setSplit(state.split - step);
      } else if (ev.key === "ArrowRight") {
        setSplit(state.split + step);
      } else if (ev.key === "Home") {
        setSplit(0);
      } else if (ev.key === "End") {
        setSplit(100);
      } else {
        return;
      }
      ev.preventDefault();
    });
  })();

  /* ------------------------------------------------------------- shortcuts */

  el.btnRun.addEventListener("click", () => {
    if (state.busy) return;
    if (!state.file) {
      el.fileInput.click();
      return;
    }
    run();
  });

  document.addEventListener("keydown", (ev) => {
    const mod = ev.metaKey || ev.ctrlKey;
    if (mod && ev.key === "Enter") {
      ev.preventDefault();
      if (!state.busy) run();
      return;
    }
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
    if (ev.key === "1") setTab("compare");
    else if (ev.key === "2") setTab("result");
    else if (ev.key === "3") setTab("mask");
    else if (ev.key === "4") setTab("diag");
    else if (ev.key.toLowerCase() === "z") setZoom(el.stage.dataset.zoom === "actual" ? "fit" : "actual");
  });

  window.addEventListener("resize", syncParamsOverflow);
  el.params.addEventListener("scroll", syncParamsOverflow, { passive: true });

  window.addEventListener("beforeunload", () => {
    for (const url of state.urls) {
      if (url.startsWith("blob:")) URL.revokeObjectURL(url);
    }
  });

  /* ------------------------------------------------------------------ init */

  (async function init() {
    setStatus("connecting", "连接中");
    const url = new URL(location.href);
    const token = url.searchParams.get("token");
    if (token) {
      url.searchParams.delete("token");
      history.replaceState(null, "", url.pathname + url.search + url.hash);
      try {
        await login(token);
        toast("令牌已接受，写入会话。", "ok");
      } catch (err) {
        toast(err.message, "error", 6000);
        requestToken("这个链接里的令牌没通过校验。");
        return;
      }
    }
    const ok = await boot();
    if (ok) {
      requestAnimationFrame(syncParamsOverflow);
      // A first-run hint that survives reloads without nagging.
      if (state.schema) {
        el.metaNote.textContent = "拖入照片，或点击虚线框选择";
      }
      const requested = url.searchParams.get("job") || lsGet("skin.lastJob");
      if (requested) {
        if (url.searchParams.has("job")) {
          url.searchParams.delete("job");
          history.replaceState(null, "", url.pathname + url.search + url.hash);
        }
        const restored = await restore(requested);
        if (restored) {
          const want = url.searchParams.get("tab");
          if (["compare", "result", "mask", "diag"].includes(want)) setTab(want);
        } else {
          lsSet("skin.lastJob", "");
        }
      }
      requestAnimationFrame(syncParamsOverflow);
    }
  })();
})();
