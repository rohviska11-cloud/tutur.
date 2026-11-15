/* forest.js — menjalankan Pipeline(SimpleImputer median -> StandardScaler -> RandomForest) hasil ekspor Colab.
 * Format biner (little-endian, berurutan): thr f32[N], left i32[N], right i32[N], value f32[L*K], feat i16[N]
 * Daun: feat = -1, left = indeks baris probabilitas daun.
 */
(function (root) {
  "use strict";

  function parse(meta, buf) {
    const N = meta.n_nodes, L = meta.n_leaves, K = meta.n_classes;
    let off = 0;
    const thr = new Float32Array(buf, off, N); off += 4 * N;
    const left = new Int32Array(buf, off, N); off += 4 * N;
    const right = new Int32Array(buf, off, N); off += 4 * N;
    const value = new Float32Array(buf, off, L * K); off += 4 * L * K;
    const feat = new Int16Array(buf, off, N);
    return { meta, thr, left, right, value, feat, roots: meta.roots, K };
  }

  async function load(baseUrl, name) {
    const meta = await (await fetch(`${baseUrl}/${name}.json`, { cache: "force-cache" })).json();
    const buf = await (await fetch(`${baseUrl}/${meta.bin}`, { cache: "force-cache" })).arrayBuffer();
    return parse(meta, buf);
  }

  // featsObj: {nama_kolom: nilai}; return {proba:[...], label, classes}
  function predict(model, featsObj) {
    const m = model.meta, F = m.feature_cols.length;
    const x = new Float32Array(F);
    for (let j = 0; j < F; j++) {
      let v = featsObj[m.feature_cols[j]];
      if (v === undefined || v === null || !Number.isFinite(v)) v = m.imputer_median[j];
      x[j] = Math.fround((v - m.scaler_mean[j]) / m.scaler_scale[j]);
    }
    const K = model.K, acc = new Float64Array(K);
    const { thr, left, right, feat, value } = model;
    for (const r of model.roots) {
      let n = r;
      while (feat[n] >= 0) n = x[feat[n]] <= thr[n] ? left[n] : right[n];
      const base = left[n] * K;
      for (let k = 0; k < K; k++) acc[k] += value[base + k];
    }
    const proba = Array.from(acc, (v) => v / model.roots.length);
    let best = 0;
    for (let k = 1; k < K; k++) if (proba[k] > proba[best]) best = k;
    return { proba, label: m.classes[best], classes: m.classes };
  }

  const Forest = { load, parse, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = Forest;
  else root.Forest = Forest;
})(typeof self !== "undefined" ? self : this);
