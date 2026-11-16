/* app.js — antarmuka: rekam/unggah, tampilan klip, kirim ke worker, render hasil. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const TARGET_SR = 16000;
  const MAX_REC = 10;

  // ---------- kecil-kecil ----------
  const toastEl = $("#toast");
  let toastT;
  function toast(msg, ms = 3600) {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toastT);
    toastT = setTimeout(() => toastEl.classList.remove("show"), ms);
  }
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : "–");
  const pct = (v) => `${Math.round(v * 100)}%`;

  // ---------- sidebar: menu HP + penanda bagian aktif ----------
  const setNav = (open) => {
    document.body.classList.toggle("nav-open", open);
    $("#burger").setAttribute("aria-expanded", open);
  };
  $("#burger").addEventListener("click", () => setNav(!document.body.classList.contains("nav-open")));
  $("#scrim").addEventListener("click", () => setNav(false));
  $$("#sideNav a").forEach((a) => a.addEventListener("click", () => setNav(false)));
  const navLinks = $$("#sideNav a");
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      let id = e.target.id === "hasil" ? "coba" : e.target.id;
      navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  ["coba", "hasil", "torgo", "metode", "kinerja", "bacaan"].forEach((id) => spy.observe(document.getElementById(id)));

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.15 });
  $$(".reveal").forEach((el) => io.observe(el));

  const meterIO = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      $$(".meter", e.target).forEach((m) => ($(".m-bar i", m).style.width = m.dataset.val + "%"));
      meterIO.unobserve(e.target);
    }
  }, { threshold: 0.4 });
  meterIO.observe($(".meters"));

  // ---------- kalimat contoh (gaya prompt TORGO) ----------
  const PROMPTS = ["The quick brown fox", "yes", "no", "Except in the winter", "pat", "air", "She had your dark suit",
    "up", "sip", "Bring me the yellow cup", "right", "Grandfather likes to be modern", "pass", "window", "eee (vokal panjang)"];
  let pIdx = 0;
  $("#shuffle").addEventListener("click", () => {
    pIdx = (pIdx + 1 + Math.floor(Math.random() * (PROMPTS.length - 1))) % PROMPTS.length;
    const el = $("#promptWord");
    el.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 350 });
    el.textContent = `"${PROMPTS[pIdx]}"`;
  });

  // ---------- tab ----------
  const tabs = $(".tabs");
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    if (recording) return;
    $$(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
    tabs.dataset.on = t.dataset.tab;
    $$(".panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== t.dataset.tab));
  }));

  // ---------- worker & model ----------
  const worker = new Worker("assets/js/worker.js");
  const modelBase = new URL("model", location.href).href;
  let modelMeta = null, reqId = 0;
  const pending = new Map();
  worker.onmessage = (ev) => {
    const d = ev.data, cb = pending.get(d.id);
    if (!cb) return;
    if (d.type === "progress") { cb.progress && cb.progress(d); return; }
    pending.delete(d.id);
    d.type === "error" ? cb.reject(new Error(d.message)) : cb.resolve(d);
  };
  const call = (msg, transfer, progress) => new Promise((resolve, reject) => {
    const id = ++reqId;
    pending.set(id, { resolve, reject, progress });
    worker.postMessage({ ...msg, id, base: modelBase }, transfer || []);
  });
  call({ type: "init" }).then((d) => {
    modelMeta = d.meta;
    updateAnalyzeBtn();
  }).catch(() => {
    toast("Model belum tersedia. Silakan muat ulang halaman beberapa saat lagi.", 6000);
  });

  // ---------- audio util ----------
  function mixToMono(buf) {
    const n = buf.length, ch = buf.numberOfChannels;
    if (ch === 1) return new Float32Array(buf.getChannelData(0));
    const out = new Float32Array(n);
    for (let c = 0; c < ch; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
    return out;
  }
  async function resampleTo16k(samples, sr) {
    if (sr === TARGET_SR) return samples;
    const len = Math.max(1, Math.round((samples.length * TARGET_SR) / sr));
    const off = new OfflineAudioContext(1, len, TARGET_SR);
    const b = off.createBuffer(1, samples.length, sr);
    b.copyToChannel(samples, 0);
    const src = off.createBufferSource();
    src.buffer = b; src.connect(off.destination); src.start();
    return (await off.startRendering()).getChannelData(0).slice();
  }
  async function decodeFile(file) {
    const ab = await file.arrayBuffer();
    // OfflineAudioContext 16 kHz: WAV 16 kHz (seperti TORGO) tidak di-resample sama sekali
    const ctx = new OfflineAudioContext(1, 1, TARGET_SR);
    const buf = await ctx.decodeAudioData(ab.slice(0));
    let origSr = buf.sampleRate;
    try { const h = new DataView(ab); if (h.getUint32(0, false) === 0x52494646) origSr = h.getUint32(24, true); } catch (e) { /* bukan wav */ }
    return { samples: mixToMono(buf), sr: buf.sampleRate, origSr, channels: buf.numberOfChannels };
  }
  // potong hening panjang di awal/akhir rekaman mikrofon, sisakan 0,3 s (rekaman TORGO juga punya jeda pendek)
  function trimEdges(y, sr) {
    const hop = Math.round(sr * 0.02), n = Math.floor(y.length / hop);
    if (n < 3) return y;
    const e = new Float32Array(n);
    let mx = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = i * hop; k < (i + 1) * hop; k++) s += y[k] * y[k];
      e[i] = Math.sqrt(s / hop); if (e[i] > mx) mx = e[i];
    }
    const thr = mx * 0.06;
    let a = 0, b = n - 1;
    while (a < n && e[a] < thr) a++;
    while (b > a && e[b] < thr) b--;
    const margin = Math.round(0.3 * sr);
    return y.slice(Math.max(0, a * hop - margin), Math.min(y.length, (b + 1) * hop + margin));
  }
  function toWav(samples, sr) {
    const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, "data"); v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
    return new Blob([buf], { type: "audio/wav" });
  }

  // ---------- gambar gelombang ----------
  function drawClip(samples) {
    const c = $("#clipWave"), g = c.getContext("2d");
    const W = (c.width = c.clientWidth * devicePixelRatio), H = (c.height = c.clientHeight * devicePixelRatio);
    g.clearRect(0, 0, W, H);
    const step = Math.max(1, Math.floor(samples.length / W));
    let peak = 1e-6;
    for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
    const bar = 3 * devicePixelRatio, gap = 2 * devicePixelRatio;
    const grad = g.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, "#1f8a83"); grad.addColorStop(1, "#0d3b40");
    g.fillStyle = grad;
    for (let x = 0; x < W; x += bar + gap) {
      const i0 = Math.floor((x / W) * samples.length);
      let m = 0;
      for (let k = i0; k < Math.min(samples.length, i0 + step * (bar + gap)); k++) m = Math.max(m, Math.abs(samples[k]));
      const h = Math.max(2 * devicePixelRatio, (m / peak) * (H * 0.92));
      g.beginPath();
      g.roundRect ? g.roundRect(x, (H - h) / 2, bar, h, bar / 2) : g.rect(x, (H - h) / 2, bar, h);
      g.fill();
    }
  }

  // ---------- klip aktif ----------
  let clip = null; // {samples (16k), name, info, url}
  const player = $("#player"), playBtn = $("#playBtn");
  function setClip(c) {
    if (clip && clip.url) URL.revokeObjectURL(clip.url);
    clip = c;
    $("#clip").classList.toggle("hidden", !c);
    if (c) {
      $("#clipName").textContent = c.name;
      $("#clipInfo").textContent = c.info;
      player.src = c.url;
      requestAnimationFrame(() => drawClip(c.samples));
    } else {
      player.removeAttribute("src");
    }
    updateAnalyzeBtn();
  }
  function updateAnalyzeBtn() { $("#analyzeBtn").disabled = !(clip && modelMeta); }
  playBtn.addEventListener("click", () => (player.paused ? player.play() : player.pause()));
  player.addEventListener("play", () => playBtn.classList.add("on"));
  player.addEventListener("pause", () => playBtn.classList.remove("on"));
  player.addEventListener("ended", () => playBtn.classList.remove("on"));
  $("#clearBtn").addEventListener("click", () => setClip(null));

  // ---------- unggah ----------
  const drop = $("#drop"), fileInput = $("#fileInput");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
  fileInput.addEventListener("change", () => fileInput.files[0] && handleFile(fileInput.files[0]));
  async function handleFile(file) {
    try {
      const d = await decodeFile(file);
      const s16 = await resampleTo16k(d.samples, d.sr);
      if (s16.length < TARGET_SR * 0.15) return toast("Durasi suara terlalu pendek. Gunakan rekaman minimal setengah detik.");
      setClip({ samples: s16, name: file.name, url: URL.createObjectURL(file),
        info: `${fmt(s16.length / TARGET_SR, 2)} s · asli ${fmt(d.origSr / 1000, 1)} kHz · ${d.channels === 1 ? "mono" : d.channels + " kanal"}` });
    } catch (e) {
      toast("File tidak dapat dibaca. Silakan gunakan format WAV atau MP3.");
    } finally { fileInput.value = ""; }
  }

  // ---------- rekam ----------
  let recording = false, actx, stream, proc, srcNode, analyser, chunks = [], recStart = 0, rafId, recTimer;
  const micBtn = $("#micBtn"), live = $("#live"), lg = live.getContext("2d");
  micBtn.addEventListener("click", () => (recording ? stopRec() : startRec()));
  async function startRec() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) {
      return toast("Akses mikrofon ditolak atau mikrofon tidak ditemukan. Anda tetap dapat mengunggah file.");
    }
    actx = new (window.AudioContext || window.webkitAudioContext)();
    srcNode = actx.createMediaStreamSource(stream);
    analyser = actx.createAnalyser(); analyser.fftSize = 2048;
    proc = actx.createScriptProcessor(4096, 1, 1);
    chunks = [];
    proc.onaudioprocess = (e) => { if (recording) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    srcNode.connect(analyser); srcNode.connect(proc);
    const mute = actx.createGain(); mute.gain.value = 0; proc.connect(mute); mute.connect(actx.destination);
    recording = true; recStart = performance.now();
    micBtn.classList.add("rec"); micBtn.setAttribute("aria-label", "Berhenti merekam");
    $("#micHint").textContent = "Perekaman berlangsung. Silakan berbicara.";
    recTimer = setTimeout(stopRec, MAX_REC * 1000);
    loopLive();
  }
  function loopLive() {
    const W = (live.width = live.clientWidth * devicePixelRatio), H = (live.height = live.clientHeight * devicePixelRatio);
    const buf = new Float32Array(analyser.fftSize);
    const draw = () => {
      if (!recording) { lg.clearRect(0, 0, W, H); return; }
      analyser.getFloatTimeDomainData(buf);
      lg.clearRect(0, 0, W, H);
      const t = (performance.now() - recStart) / 1000;
      $("#timer").textContent = `${t.toFixed(1).padStart(4, "0")} s`;
      lg.lineWidth = 2 * devicePixelRatio;
      for (let pass = 0; pass < 2; pass++) {
        lg.strokeStyle = pass ? "rgba(13,59,64,.85)" : "rgba(63,179,163,.45)";
        lg.beginPath();
        for (let x = 0; x <= W; x += 3) {
          const i = Math.floor((x / W) * buf.length);
          const env = Math.sin((Math.PI * x) / W);
          const y = H / 2 + buf[i] * H * 1.6 * env * (pass ? 1 : -0.7);
          x ? lg.lineTo(x, y) : lg.moveTo(x, y);
        }
        lg.stroke();
      }
      rafId = requestAnimationFrame(draw);
    };
    draw();
  }
  async function stopRec() {
    if (!recording) return;
    recording = false;
    clearTimeout(recTimer); cancelAnimationFrame(rafId);
    micBtn.classList.remove("rec"); micBtn.setAttribute("aria-label", "Mulai merekam");
    $("#micHint").textContent = "Tekan tombol untuk mulai merekam, tekan sekali lagi untuk berhenti. Maksimal 10 detik.";
    stream.getTracks().forEach((t) => t.stop());
    proc.disconnect(); srcNode.disconnect();
    const sr = actx.sampleRate;
    await actx.close();
    const n = chunks.reduce((a, c) => a + c.length, 0);
    const raw = new Float32Array(n);
    let o = 0; for (const c of chunks) { raw.set(c, o); o += c.length; }
    lg.clearRect(0, 0, live.width, live.height);
    if (n < sr * 0.4) { $("#timer").textContent = "00.0 s"; return toast("Rekaman terlalu singkat. Silakan ulangi."); }
    const s16 = trimEdges(await resampleTo16k(raw, sr), TARGET_SR);
    setClip({ samples: s16, name: "rekaman mikrofon", url: URL.createObjectURL(toWav(s16, TARGET_SR)),
      info: `${fmt(s16.length / TARGET_SR, 2)} s · direkam ${Math.round(sr / 1000)} kHz, diolah 16 kHz` });
  }

  // ---------- analisis ----------
  const SEV_ORDER = ["typical", "mild", "moderate", "moderate-severe", "severe"];
  const SEV_ID = { typical: "Tipikal", mild: "Ringan", moderate: "Sedang", "moderate-severe": "Sedang-berat", severe: "Berat" };
  const sevRank = (lbl) => Math.min(...lbl.split("+").map((s) => { const i = SEV_ORDER.indexOf(s); return i < 0 ? 99 : i; }));
  const sevName = (lbl) => lbl.split("+").map((s) => SEV_ID[s] || s).join(" / ");

  $("#analyzeBtn").addEventListener("click", async () => {
    if (!clip) return;
    const btn = $("#analyzeBtn");
    btn.disabled = true; btn.classList.add("busy");
    const label = $(".cta-label", btn), fillEl = $("#ctaFill");
    label.textContent = "Memproses...";
    const onProgress = (p) => {
      if (p.stage === "unduh") {
        const r = p.loaded / p.total;
        fillEl.style.width = (r * 100).toFixed(1) + "%";
        label.textContent = `Mengunduh model ${Math.round(r * 100)}% (${Math.round(p.loaded / 1e6)}/${Math.round(p.total / 1e6)} MB)`;
      } else if (p.stage === "siapkan") {
        label.textContent = "Menyiapkan model...";
      } else if (p.stage === "hitung") {
        fillEl.style.width = "0";
        label.textContent = "Menganalisis suara...";
      }
    };
    try {
      const samples = clip.samples.slice();
      const res = await call({ type: "analyze", samples, sr: TARGET_SR }, [samples.buffer], onProgress);
      render(res);
    } catch (e) {
      toast("Terjadi kesalahan saat memproses: " + e.message, 6000);
    } finally {
      fillEl.style.width = "0";
      btn.classList.remove("busy");
      $(".cta-label", btn).textContent = "Klasifikasikan";
      updateAnalyzeBtn();
    }
  });

  function render(res) {
    const sec = $("#hasil");
    sec.classList.remove("hidden", "show");
    // status
    const sc = res.status.classes, sp = res.status.proba;
    const pD = sp[sc.indexOf("dysarthria")] || 0, pH = sp[sc.indexOf("healthy")] || 0;
    const fill = $("#gFill");
    fill.style.strokeDashoffset = 515.2;
    fill.classList.toggle("high", pD >= 0.5);
    $("#gPct").textContent = pct(pD);
    $("#pHealthy").textContent = pct(pH); $("#pDys").textContent = pct(pD);
    const isD = res.status.label === "dysarthria";
    const conf = Math.max(pD, pH);
    const yakin = conf >= 0.75 ? "tinggi" : conf >= 0.6 ? "sedang" : "rendah";
    $("#statusVerdict").textContent = `Terklasifikasi ${isD ? "dysarthria" : "sehat"} (keyakinan ${yakin}).`;
    // severity
    const order = res.severity.classes.map((c, i) => ({ c, p: res.severity.proba[i] })).sort((a, b) => sevRank(a.c) - sevRank(b.c));
    const best = res.severity.label;
    const lad = $("#ladder");
    lad.innerHTML = order.map((o) => `<li class="${o.c === best ? "top" : ""}"><span>${sevName(o.c)}</span><div class="lb"><i data-w="${(o.p * 100).toFixed(1)}"></i></div><em>${pct(o.p)}</em></li>`).join("");
    $("#sevVerdict").textContent = `Tingkat: ${sevName(best)}`;
    const typ = order.find((o) => o.c.split("+").includes("typical"));
    const mismatch = (isD && best.split("+").includes("typical")) || (!isD && !best.split("+").includes("typical"));
    $("#sevNote").textContent = mismatch
      ? "Kedua model dilatih secara terpisah sehingga hasilnya dapat berbeda. Hasil ini bersifat indikatif."
      : typ ? `Probabilitas tipikal ${pct(typ.p)}.` : "";
    // ciri
    const f = res.feats;
    const items = [
      ["Durasi", fmt(res.duration, 2), "s"],
      ["F0 rata-rata", fmt(f.f0_mean, 0), "Hz"],
      ["Rasio bersuara", Number.isFinite(f.voiced_ratio) ? Math.round(f.voiced_ratio * 100) : "–", "%"],
      ["HNR", fmt(f.hnr_mean, 1), "dB"],
      ["Jitter", fmt(f.jitter_local_pct, 2), "%"],
      ["Shimmer", fmt(f.shimmer_local_pct, 1), "%"],
    ];
    $("#featList").innerHTML = items.map(([k, v, u]) => `<div><dt>${k}</dt><dd>${v}<small>${u}</small></dd></div>`).join("");
    $("#resultTitle").innerHTML = isD ? "Suara menunjukkan pola <em>dysarthria</em>." : "Suara menunjukkan pola bicara <em>sehat</em>.";
    requestAnimationFrame(() => {
      sec.classList.add("show");
      sec.scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => {
        fill.style.strokeDashoffset = 515.2 * (1 - pD);
        $("#sbHealthy").style.width = pH * 100 + "%";
        $("#sbDys").style.width = pD * 100 + "%";
        $$("#ladder .lb i").forEach((i) => (i.style.width = i.dataset.w + "%"));
      }, 250);
    });
  }
})();
