/* features.js — 66 fitur per rekaman, urutan kolom mengikuti notebook (MFCC_COLS + ACOUSTIC_USED). */
(function (root) {
  "use strict";
  const DSP = root.DSP || (typeof require !== "undefined" ? require("./dsp.js") : null);
  const Praat = root.Praat || (typeof require !== "undefined" ? require("./praat.js") : null);

  const STATS = ["mean", "std", "min", "max"];
  const MFCC_COLS = [];
  for (const s of STATS) for (let i = 1; i <= 13; i++) MFCC_COLS.push(`mfcc${i}_${s}`);

  // y: Float32Array mono pada 16 kHz (belum dinormalisasi)
  function extract(yRaw, sr) {
    const y = DSP.preprocess(yRaw, 0.95);
    const feats = {};
    const seq = DSP.mfcc(y, sr);
    const st = DSP.mfccStats(seq);
    MFCC_COLS.forEach((c, i) => (feats[c] = st[i]));
    Object.assign(feats, DSP.librosaFeatures(y, sr));
    Object.assign(feats, Praat.praatFeatures(y, sr, 75, 500));
    return { feats, y, duration: y.length / sr };
  }

  const Features = { extract, MFCC_COLS };
  if (typeof module !== "undefined" && module.exports) module.exports = Features;
  else root.Features = Features;
})(typeof self !== "undefined" ? self : this);
