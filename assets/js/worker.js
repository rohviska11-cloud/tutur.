/* worker.js — ekstraksi fitur + prediksi di thread terpisah agar animasi halaman tetap mulus. */
importScripts("dsp.js", "praat.js", "features.js", "forest.js");

let models = null;

async function ensureModels(base) {
  if (models) return models;
  const [status, severity] = await Promise.all([Forest.load(base, "speech_status"), Forest.load(base, "severity")]);
  models = { status, severity };
  return models;
}

self.onmessage = async (ev) => {
  const { type, id } = ev.data;
  try {
    if (type === "init") {
      const m = await ensureModels(ev.data.base);
      self.postMessage({ id, type: "ready", meta: { status: m.status.meta, severity: m.severity.meta } });
    } else if (type === "analyze") {
      const m = await ensureModels(ev.data.base);
      const t0 = performance.now();
      const { feats, duration } = Features.extract(ev.data.samples, ev.data.sr);
      const status = Forest.predict(m.status, feats);
      const severity = Forest.predict(m.severity, feats);
      self.postMessage({ id, type: "result", feats, duration, status, severity, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ id, type: "error", message: String((err && err.message) || err) });
  }
};
