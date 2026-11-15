/* dsp.js
 * Replika fitur librosa 0.10 yang dipakai di notebook:
 *   - preprocess_audio (mono, DC offset, normalisasi puncak 0.95, tanpa trimming)
 *   - librosa.feature.mfcc (n_mfcc=13, n_fft=512, win_length=400, hop=160, 128 mel slaney, top_db=80)
 *   - zero_crossing_rate, rms (frame 400, hop 160), spectral_centroid (n_fft 512, hop 160)
 * Bisa dipakai di browser, Web Worker, maupun Node.
 */
(function (root) {
  "use strict";

  // ---------- FFT radix-2 (in place) ----------
  const fftCache = {};
  function fftTables(n) {
    if (fftCache[n]) return fftCache[n];
    const rev = new Uint32Array(n);
    const bits = Math.log2(n) | 0;
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / n);
      sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
    return (fftCache[n] = { rev, cos, sin });
  }

  function fft(re, im) {
    const n = re.length;
    const { rev, cos, sin } = fftTables(n);
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = cos[k * step], wi = sin[k * step];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  // ---------- preprocessing (identik dengan preprocess_audio di notebook) ----------
  function preprocess(y, peakTarget) {
    peakTarget = peakTarget || 0.95;
    const n = y.length;
    const out = new Float32Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = Number.isFinite(y[i]) ? y[i] : 0;
      out[i] = v; sum += v;
    }
    const mean = n ? sum / n : 0;
    let peak = 0;
    for (let i = 0; i < n; i++) {
      out[i] = out[i] - mean;
      const a = Math.abs(out[i]);
      if (a > peak) peak = a;
    }
    if (peak > 1e-6) {
      const g = peakTarget / peak;
      for (let i = 0; i < n; i++) out[i] = out[i] * g;
    }
    return out;
  }

  // ---------- window & framing ----------
  function hannPeriodic(len) {
    const w = new Float64Array(len);
    for (let i = 0; i < len; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / len);
    return w;
  }
  function padCenter(w, size) {
    const out = new Float64Array(size);
    const l = Math.floor((size - w.length) / 2);
    out.set(w, l);
    return out;
  }

  // |STFT|^power, center=True, pad_mode="constant"; return array of Float64Array(nfft/2+1)
  function stft(y, nfft, hop, winLength, power) {
    const win = padCenter(hannPeriodic(winLength), nfft);
    const pad = nfft >> 1;
    const n = y.length;
    const nFrames = 1 + Math.floor(n / hop);
    const nBins = (nfft >> 1) + 1;
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    const frames = new Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      const start = t * hop - pad;
      for (let k = 0; k < nfft; k++) {
        const idx = start + k;
        re[k] = idx >= 0 && idx < n ? y[idx] * win[k] : 0;
        im[k] = 0;
      }
      fft(re, im);
      const mag = new Float64Array(nBins);
      for (let k = 0; k < nBins; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        mag[k] = power === 2 ? p : Math.sqrt(p);
      }
      frames[t] = mag;
    }
    return frames;
  }

  // ---------- mel filterbank (librosa.filters.mel, htk=False, norm="slaney") ----------
  function hzToMel(f) {
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
    return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
  }
  function melToHz(m) {
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
    return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
  }
  const melCache = {};
  function melFilterbank(sr, nfft, nMels) {
    const key = sr + "_" + nfft + "_" + nMels;
    if (melCache[key]) return melCache[key];
    const nBins = (nfft >> 1) + 1;
    const fftFreqs = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) fftFreqs[k] = (k * (sr / 2)) / (nBins - 1);
    const mMin = hzToMel(0), mMax = hzToMel(sr / 2);
    const melF = new Float64Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) melF[i] = melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1));
    const weights = [];
    for (let i = 0; i < nMels; i++) {
      const w = new Float64Array(nBins);
      const fd0 = melF[i + 1] - melF[i], fd1 = melF[i + 2] - melF[i + 1];
      const enorm = 2 / (melF[i + 2] - melF[i]);
      for (let k = 0; k < nBins; k++) {
        const lower = -(melF[i] - fftFreqs[k]) / fd0;
        const upper = (melF[i + 2] - fftFreqs[k]) / fd1;
        w[k] = Math.max(0, Math.min(lower, upper)) * enorm;
      }
      weights.push(w);
    }
    return (melCache[key] = weights);
  }

  // ---------- MFCC ----------
  function mfcc(y, sr, opt) {
    opt = Object.assign({ nMfcc: 13, nfft: 512, winLength: 400, hop: 160, nMels: 128, topDb: 80 }, opt || {});
    const spec = stft(y, opt.nfft, opt.hop, opt.winLength, 2);
    const fb = melFilterbank(sr, opt.nfft, opt.nMels);
    const T = spec.length, M = opt.nMels;
    const logMel = new Array(T);
    let gmax = -Infinity;
    for (let t = 0; t < T; t++) {
      const row = new Float64Array(M);
      const s = spec[t];
      for (let m = 0; m < M; m++) {
        const w = fb[m];
        let acc = 0;
        for (let k = 0; k < s.length; k++) acc += w[k] * s[k];
        const db = 10 * Math.log10(Math.max(1e-10, acc));
        row[m] = db;
        if (db > gmax) gmax = db;
      }
      logMel[t] = row;
    }
    const floor = gmax - opt.topDb;
    // DCT-II ortho
    const C = opt.nMfcc;
    const dct = [];
    for (let k = 0; k < C; k++) {
      const row = new Float64Array(M);
      const scale = k === 0 ? Math.sqrt(1 / M) : Math.sqrt(2 / M);
      for (let n = 0; n < M; n++) row[n] = scale * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * M));
      dct.push(row);
    }
    const out = new Array(T);
    for (let t = 0; t < T; t++) {
      const x = logMel[t];
      for (let m = 0; m < M; m++) if (x[m] < floor) x[m] = floor;
      const c = new Float64Array(C);
      for (let k = 0; k < C; k++) {
        let acc = 0;
        const d = dct[k];
        for (let m = 0; m < M; m++) acc += d[m] * x[m];
        c[k] = acc;
      }
      out[t] = c;
    }
    return out; // T x 13
  }

  function meanStd(arr) {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    const mean = s / arr.length;
    let v = 0;
    for (let i = 0; i < arr.length; i++) v += (arr[i] - mean) * (arr[i] - mean);
    return [mean, Math.sqrt(v / arr.length)];
  }

  // 52 statistik: mean(13), std(13), min(13), max(13)
  function mfccStats(seq) {
    const C = seq[0].length, T = seq.length;
    const mean = [], std = [], mn = [], mx = [];
    for (let c = 0; c < C; c++) {
      const col = new Float64Array(T);
      let lo = Infinity, hi = -Infinity;
      for (let t = 0; t < T; t++) {
        col[t] = seq[t][c];
        if (col[t] < lo) lo = col[t];
        if (col[t] > hi) hi = col[t];
      }
      const [m, s] = meanStd(col);
      mean.push(m); std.push(s); mn.push(lo); mx.push(hi);
    }
    return mean.concat(std, mn, mx);
  }

  // ---------- ZCR, RMS, centroid ----------
  function zcr(y, frameLength, hop) {
    const pad = frameLength >> 1, n = y.length;
    const nFrames = 1 + Math.floor(n / hop);
    const val = (i) => {
      let j = i - pad;
      if (j < 0) j = 0;
      if (j >= n) j = n - 1; // mode="edge"
      const v = y[j];
      return Math.abs(v) <= 1e-10 ? 0 : v;
    };
    const out = new Float64Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      const s = t * hop;
      let prev = val(s) < 0, count = 0;
      for (let k = 1; k < frameLength; k++) {
        const cur = val(s + k) < 0;
        if (cur !== prev) count++;
        prev = cur;
      }
      out[t] = count / frameLength;
    }
    return out;
  }

  function rms(y, frameLength, hop) {
    const pad = frameLength >> 1, n = y.length;
    const nFrames = 1 + Math.floor(n / hop);
    const out = new Float64Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      const s = t * hop - pad;
      let acc = 0;
      for (let k = 0; k < frameLength; k++) {
        const j = s + k;
        if (j >= 0 && j < n) acc += y[j] * y[j];
      }
      out[t] = Math.sqrt(acc / frameLength);
    }
    return out;
  }

  function centroid(y, sr, nfft, hop) {
    const spec = stft(y, nfft, hop, nfft, 1);
    const nBins = (nfft >> 1) + 1;
    const out = new Float64Array(spec.length);
    for (let t = 0; t < spec.length; t++) {
      const s = spec[t];
      let tot = 0, acc = 0;
      for (let k = 0; k < nBins; k++) tot += s[k];
      if (tot < 1.1754944e-38) tot = 1;
      for (let k = 0; k < nBins; k++) acc += ((k * (sr / 2)) / (nBins - 1)) * (s[k] / tot);
      out[t] = acc;
    }
    return out;
  }

  function librosaFeatures(y, sr) {
    const z = meanStd(zcr(y, 400, 160));
    const r = meanStd(rms(y, 400, 160));
    const c = meanStd(centroid(y, sr, 512, 160));
    return { zcr_mean: z[0], zcr_std: z[1], rms_mean: r[0], rms_std: r[1], centroid_mean: c[0], centroid_std: c[1] };
  }

  const DSP = { fft, preprocess, stft, mfcc, mfccStats, zcr, rms, centroid, librosaFeatures, meanStd };
  if (typeof module !== "undefined" && module.exports) module.exports = DSP;
  else root.DSP = DSP;
})(typeof self !== "undefined" ? self : this);
