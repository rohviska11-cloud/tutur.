/* praat.js
 * Port JavaScript dari algoritma Praat yang dipanggil notebook lewat parselmouth:
 *   snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)          -> AC Hanning
 *   "To PointProcess (periodic, cc)" 75 500                                  -> pulsa glotal
 *   "Get jitter (local)" 0 0 0.0001 0.02 1.3
 *   "Get shimmer (local)" 0 0 0.0001 0.02 1.3 1.6
 *   "To Harmonicity (cc)" 0.01 75 0.1 1.0 lalu "Get mean"                    -> HNR
 * Mengikuti kode sumber Praat (fon/Sound_to_Pitch.cpp, Pitch.cpp, Pitch_to_PointProcess.cpp,
 * VoiceAnalysis.cpp, AmplitudeTier.cpp, melder/NUMinterpol.cpp). Indeks 1-based Praat dipertahankan
 * lewat helper agar mudah dicocokkan dengan sumber aslinya.
 */
(function (root) {
  "use strict";
  const PI = Math.PI;
  const AC_HANNING = 0, FCC_ACCURATE = 3;
  const DEPTH_SINC70 = 70, DEPTH_SINC700 = 700;

  // ---------- Sound (konvensi Sampled Praat) ----------
  function makeSound(y, sr) {
    const nx = y.length, dx = 1.0 / sr;
    const z = new Float64Array(nx + 2); // z[1..nx]
    for (let i = 0; i < nx; i++) z[i + 1] = y[i];
    return { z, nx, dx, x1: 0.5 * dx, xmin: 0, xmax: nx * dx };
  }
  const xToLowIndex = (s, x) => Math.floor((x - s.x1) / s.dx) + 1;
  const xToHighIndex = (s, x) => Math.ceil((x - s.x1) / s.dx) + 1;
  const xToNearestIndex = (s, x) => Math.floor((x - s.x1) / s.dx + 0.5) + 1;
  const indexToX = (s, i) => s.x1 + (i - 1) * s.dx;

  // ---------- NUM_interpolate_sinc (jendela raised-cosine, versi parselmouth) ----------
  // y: Float64Array 1-based (y[1..size]), size: jumlah elemen
  function interpolateSinc(y, size, x, maxDepth) {
    const midleft = Math.floor(x), midright = midleft + 1;
    if (size < 1) return NaN;
    if (x < 1) return y[1];
    if (x > size) return y[size];
    if (x === midleft) return y[midleft];
    if (maxDepth > midright - 1) maxDepth = midright - 1;
    if (maxDepth > size - midleft) maxDepth = size - midleft;
    if (maxDepth <= 0) return y[Math.floor(x + 0.5)];
    if (maxDepth === 1) return y[midleft] + (x - midleft) * (y[midright] - y[midleft]);
    if (maxDepth === 2) {
      const yl = y[midleft], yr = y[midright];
      const dyl = 0.5 * (yr - y[midleft - 1]), dyr = 0.5 * (y[midright + 1] - yl);
      const fil = x - midleft, fir = midright - x;
      return yl * fir + yr * fil - fil * fir * (0.5 * (dyr - dyl) + (fil - 0.5) * (dyl + dyr - 2 * (yr - yl)));
    }
    const left = midright - maxDepth, right = midleft + maxDepth;
    let result = 0;
    let a = PI * (x - midleft);
    let halfsina = 0.5 * Math.sin(a);
    let aa = a / (x - left + 1), daa = PI / (x - left + 1);
    let cosaa = Math.cos(aa), sinaa = Math.sin(aa), cosdaa = Math.cos(daa), sindaa = Math.sin(daa);
    for (let ix = midleft; ix >= left; ix--) {
      const d = (halfsina / a) * (1.0 + cosaa);
      result += y[ix] * d;
      a += PI;
      const help = cosaa * cosdaa - sinaa * sindaa;
      sinaa = cosaa * sindaa + sinaa * cosdaa;
      cosaa = help;
      halfsina = -halfsina;
    }
    a = PI * (midright - x);
    halfsina = 0.5 * Math.sin(a);
    aa = a / (right - x + 1); daa = PI / (right - x + 1);
    cosaa = Math.cos(aa); sinaa = Math.sin(aa); cosdaa = Math.cos(daa); sindaa = Math.sin(daa);
    for (let ix = midright; ix <= right; ix++) {
      const d = (halfsina / a) * (1.0 + cosaa);
      result += y[ix] * d;
      a += PI;
      const help = cosaa * cosdaa - sinaa * sindaa;
      sinaa = cosaa * sindaa + sinaa * cosdaa;
      cosaa = help;
      halfsina = -halfsina;
    }
    return result;
  }

  // ---------- NUMminimize_brent ----------
  function brentMinimize(f, a, b, tol) {
    const golden = 1 - 0.6180339887498948482;
    const sqrtEps = Math.sqrt(2.220446049250313e-16);
    let v = a + golden * (b - a), fv = f(v);
    let x = v, w = v, fx = fv, fw = fv;
    for (let iter = 1; iter <= 60; iter++) {
      const mid = (a + b) / 2;
      const tolAct = sqrtEps * Math.abs(x) + tol / 3;
      if (Math.abs(x - mid) + (b - a) / 2 <= 2 * tolAct) return [x, fx];
      let step = golden * (x < mid ? b - x : a - x);
      if (Math.abs(x - w) >= tolAct) {
        const t = (x - w) * (fx - fv);
        let q = (x - v) * (fx - fw);
        let p = (x - v) * q - (x - w) * t;
        q = 2 * (q - t);
        if (q > 0) p = -p; else q = -q;
        if (Math.abs(p) < Math.abs(step * q) && p > q * (a - x + 2 * tolAct) && p < q * (b - x - 2 * tolAct)) step = p / q;
      }
      if (Math.abs(step) < tolAct) step = step > 0 ? tolAct : -tolAct;
      const t = x + step, ft = f(t);
      if (ft <= fx) {
        if (t < x) b = x; else a = x;
        v = w; w = x; x = t;
        fv = fw; fw = fx; fx = ft;
      } else {
        if (t < x) a = t; else b = t;
        if (ft <= fw || w === x) { v = w; w = t; fv = fw; fw = ft; }
        else if (ft <= fv || v === x || v === w) { v = t; fv = ft; }
      }
    }
    return [x, fx];
  }

  function improveMaximum(y, size, ixmid, depth) {
    if (ixmid <= 1) return [1, y[1]];
    if (ixmid >= size) return [size, y[size]];
    const [xr, fr] = brentMinimize((xx) => -interpolateSinc(y, size, xx, depth), ixmid - 1, ixmid + 1, 1e-10);
    return [xr, -fr];
  }

  // ---------- FFT real (pakai DSP.fft) untuk autokorelasi ----------
  function powerAutocorr(frame, nfft, fft) {
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    re.set(frame.subarray(0, nfft));
    fft(re, im);
    for (let k = 0; k < nfft; k++) { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0; }
    fft(re, im); // spektrum daya simetris -> FFT maju = IFFT * n (skala tidak berpengaruh, dinormalisasi)
    return re; // re[lag]
  }

  // ---------- Sound_to_Pitch_any ----------
  function toPitch(snd, o) {
    const fft = (root.DSP || (typeof require !== "undefined" ? require("./dsp.js") : null)).fft;
    let { method, periodsPerWindow, dt, pitchFloor, pitchCeiling, maxnCandidates } = o;
    const { silenceThreshold, voicingThreshold, octaveCost, octaveJumpCost, voicedUnvoicedCost } = o;
    if (maxnCandidates < pitchCeiling / pitchFloor) maxnCandidates = Math.floor(pitchCeiling / pitchFloor);
    if (dt <= 0) dt = periodsPerWindow / pitchFloor / 4.0;
    let brentDepth, interpolationDepth;
    if (method === AC_HANNING) { brentDepth = DEPTH_SINC70; interpolationDepth = 0.5; }
    else { brentDepth = DEPTH_SINC700; interpolationDepth = 1.0; } // FCC_ACCURATE
    const duration = snd.dx * snd.nx;
    if (pitchFloor < periodsPerWindow / duration) throw new Error("sound too short");
    const nsampPeriod = Math.floor(1.0 / snd.dx / pitchFloor);
    const halfnsampPeriod = Math.floor(nsampPeriod / 2) + 1;
    if (pitchCeiling > 0.5 / snd.dx) pitchCeiling = 0.5 / snd.dx;
    const dtWindow = periodsPerWindow / pitchFloor;
    let nsampWindow = Math.floor(dtWindow / snd.dx);
    const halfnsampWindow = Math.floor(nsampWindow / 2) - 1;
    if (halfnsampWindow < 2) throw new Error("window too short");
    nsampWindow = halfnsampWindow * 2;
    const maximumLag = Math.min(Math.floor(nsampWindow / periodsPerWindow) + 2, nsampWindow);

    // Sampled_shortTermAnalysis
    const windowDuration = method >= 2 ? 1.0 / pitchFloor + dtWindow : dtWindow;
    if (windowDuration > duration) throw new Error("zero frames");
    const nFrames = Math.floor((duration - windowDuration) / dt) + 1;
    if (nFrames < 1) throw new Error("zero frames");
    const ourMidTime = snd.x1 - 0.5 * snd.dx + 0.5 * duration;
    const t1 = ourMidTime - 0.5 * nFrames * dt + 0.5 * dt;
    const pitch = { nx: nFrames, dx: dt, x1: t1, xmin: snd.xmin, xmax: snd.xmax, ceiling: pitchCeiling, frames: new Array(nFrames) };
    for (let i = 0; i < nFrames; i++) pitch.frames[i] = { intensity: 0, cand: [{ f: 0, s: 0 }] };

    // global peak
    let mean = 0;
    for (let i = 1; i <= snd.nx; i++) mean += snd.z[i];
    mean /= snd.nx;
    let globalPeak = 0;
    for (let i = 1; i <= snd.nx; i++) { const v = Math.abs(snd.z[i] - mean); if (v > globalPeak) globalPeak = v; }
    if (globalPeak === 0) return pitch;

    let nsampFFT = 0, brentIxmax, window = null, windowR = null;
    if (method >= 2) {
      brentIxmax = Math.floor(nsampWindow * interpolationDepth);
    } else {
      nsampFFT = 1;
      while (nsampFFT < nsampWindow * (1 + interpolationDepth)) nsampFFT *= 2;
      window = new Float64Array(nsampWindow + 1);
      for (let i = 1; i <= nsampWindow; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * PI * i) / (nsampWindow + 1));
      const wf = new Float64Array(nsampFFT);
      for (let i = 1; i <= nsampWindow; i++) wf[i - 1] = window[i];
      const ac = powerAutocorr(wf, nsampFFT, fft);
      windowR = new Float64Array(nsampFFT + 1); // 1-based: windowR[lag+1]
      for (let i = 1; i <= nsampFFT; i++) windowR[i] = ac[i - 1] / ac[0];
      brentIxmax = Math.floor(nsampWindow * interpolationDepth);
    }

    const B = brentIxmax;
    const rsize = 2 * B + 1; // y[1..rsize]; lag L <-> index L + B + 1
    const frame = new Float64Array(Math.max(nsampFFT, nsampWindow) + 1);
    const z = snd.z;

    for (let iframe = 1; iframe <= nFrames; iframe++) {
      const pf = pitch.frames[iframe - 1];
      const t = t1 + (iframe - 1) * dt;
      const leftSample = xToLowIndex(snd, t), rightSample = leftSample + 1;
      // local mean
      let startSample = rightSample - nsampPeriod, endSample = leftSample + nsampPeriod;
      let localMean = 0;
      for (let i = startSample; i <= endSample; i++) localMean += (i >= 1 && i <= snd.nx) ? z[i] : 0;
      localMean /= 2 * nsampPeriod;
      // window copy
      startSample = rightSample - halfnsampWindow;
      if (method < 2) {
        for (let j = 1, i = startSample; j <= nsampWindow; j++, i++) frame[j] = ((i >= 1 && i <= snd.nx) ? z[i] : 0) - localMean, frame[j] *= window[j];
      } else {
        for (let j = 1, i = startSample; j <= nsampWindow; j++, i++) frame[j] = ((i >= 1 && i <= snd.nx) ? z[i] : 0) - localMean;
      }
      // local peak
      let ss = halfnsampWindow + 1 - halfnsampPeriod; if (ss < 1) ss = 1;
      let ee = halfnsampWindow + halfnsampPeriod; if (ee > nsampWindow) ee = nsampWindow;
      let localPeak = 0;
      for (let j = ss; j <= ee; j++) { const v = Math.abs(frame[j]); if (v > localPeak) localPeak = v; }
      pf.intensity = localPeak > globalPeak ? 1.0 : localPeak / globalPeak;

      // correlation r (y 1-based)
      const r = new Float64Array(rsize + 2);
      const R = (lag) => r[lag + B + 1];
      if (method >= 2) {
        const startTime = t - 0.5 * (1.0 / pitchFloor + dtWindow);
        let localSpan = maximumLag + nsampWindow;
        let st = xToLowIndex(snd, startTime); if (st < 1) st = 1;
        if (localSpan > snd.nx + 1 - st) localSpan = snd.nx + 1 - st;
        const localMaximumLag = localSpan - nsampWindow;
        const offset = st - 1;
        let sumx2 = 0;
        for (let i = 1; i <= nsampWindow; i++) { const x = z[offset + i] - localMean; sumx2 += x * x; }
        let sumy2 = sumx2;
        r[B + 1] = 1.0;
        for (let i = 1; i <= localMaximumLag; i++) {
          let product = 0;
          const y0 = z[offset + i] - localMean;
          const yZ = z[offset + i + nsampWindow] - localMean;
          sumy2 += yZ * yZ - y0 * y0;
          for (let j = 1; j <= nsampWindow; j++) product += (z[offset + j] - localMean) * (z[offset + i + j] - localMean);
          const x2y2 = sumx2 * sumy2;
          const val = x2y2 <= 0 ? 0 : product / Math.sqrt(x2y2);
          if (i <= B) { r[i + B + 1] = val; r[-i + B + 1] = val; }
        }
      } else {
        const fr = new Float64Array(nsampFFT);
        for (let j = 1; j <= nsampWindow; j++) fr[j - 1] = frame[j];
        const ac = powerAutocorr(fr, nsampFFT, fft);
        r[B + 1] = 1.0;
        for (let i = 1; i <= B; i++) {
          const val = ac[i] / (ac[0] * windowR[i + 1]);
          r[i + B + 1] = val; r[-i + B + 1] = val;
        }
      }

      pf.cand = [{ f: 0, s: 0 }];
      if (localPeak === 0) continue;
      const imax = [0];
      for (let i = 2; i < maximumLag && i < B; i++) {
        const ri = R(i), rm = R(i - 1), rp = R(i + 1);
        if (ri > 0.5 * voicingThreshold && ri > rm && ri >= rp) {
          let place = 0;
          const dr = 0.5 * (rp - rm);
          const d2r = (ri - rm) + (ri - rp);
          const freq = 1.0 / snd.dx / (i + dr / d2r);
          let strength = interpolateSinc(r, rsize, 1.0 / snd.dx / freq + B + 1, 30);
          if (strength > 1.0) strength = 1.0 / strength;
          if (pf.cand.length < maxnCandidates) {
            pf.cand.push({ f: 0, s: 0 });
            place = pf.cand.length; // 1-based
          } else {
            let weakest = 2.0;
            for (let iw = 2; iw <= maxnCandidates; iw++) {
              const c = pf.cand[iw - 1];
              const ls = c.s - octaveCost * Math.log2(pitchFloor / c.f);
              if (ls < weakest) { weakest = ls; place = iw; }
            }
            if (strength - octaveCost * Math.log2(pitchFloor / freq) <= weakest) place = 0;
          }
          if (place) {
            pf.cand[place - 1].f = freq;
            pf.cand[place - 1].s = strength;
            imax[place - 1] = i;
          }
        }
      }
      for (let i = 2; i <= pf.cand.length; i++) {
        const c = pf.cand[i - 1];
        if (method !== AC_HANNING || c.f > 0) {
          const depth = c.f > 0.3 / snd.dx ? DEPTH_SINC700 : brentDepth;
          let [xmid, ymid] = improveMaximum(r, rsize, imax[i - 1] + B + 1, depth);
          xmid -= B + 1;
          c.f = 1.0 / snd.dx / xmid;
          if (ymid > 1.0) ymid = 1.0 / ymid;
          c.s = ymid;
        }
      }
    }
    pathFinder(pitch, silenceThreshold, voicingThreshold, octaveCost, octaveJumpCost, voicedUnvoicedCost, pitchCeiling);
    return pitch;
  }

  const isVoicedF = (f, ceiling) => f > 0 && f < ceiling;

  function pathFinder(p, silenceThreshold, voicingThreshold, octaveCost, octaveJumpCost, vuCost, ceiling) {
    const corr = 0.01 / p.dx;
    octaveJumpCost *= corr; vuCost *= corr;
    p.ceiling = ceiling;
    const nx = p.nx;
    const delta = new Array(nx), psi = new Array(nx);
    for (let it = 0; it < nx; it++) {
      const fr = p.frames[it];
      let uv = silenceThreshold <= 0 ? 0 : 2.0 - fr.intensity / (silenceThreshold / (1.0 + voicingThreshold));
      uv = voicingThreshold + Math.max(0, uv);
      const d = new Float64Array(fr.cand.length);
      for (let ic = 0; ic < fr.cand.length; ic++) {
        const c = fr.cand[ic];
        d[ic] = !isVoicedF(c.f, ceiling) ? uv : c.s - octaveCost * Math.log2(ceiling / c.f);
      }
      delta[it] = d;
      psi[it] = new Int32Array(fr.cand.length);
    }
    for (let it = 1; it < nx; it++) {
      const prev = p.frames[it - 1], cur = p.frames[it];
      const pd = delta[it - 1], cd = delta[it];
      for (let i2 = 0; i2 < cur.cand.length; i2++) {
        const f2 = cur.cand[i2].f;
        let maximum = -1e30, place = 0;
        const curUV = !isVoicedF(f2, ceiling);
        for (let i1 = 0; i1 < prev.cand.length; i1++) {
          const f1 = prev.cand[i1].f;
          const prevUV = !isVoicedF(f1, ceiling);
          let tc;
          if (curUV) tc = prevUV ? 0 : vuCost;
          else tc = prevUV ? vuCost : octaveJumpCost * Math.abs(Math.log2(f1 / f2));
          const value = pd[i1] - tc + cd[i2];
          if (value > maximum) { maximum = value; place = i1; }
        }
        cd[i2] = maximum;
        psi[it][i2] = place;
      }
    }
    let place = 0, maximum = delta[nx - 1][0];
    for (let ic = 1; ic < p.frames[nx - 1].cand.length; ic++) if (delta[nx - 1][ic] > maximum) { place = ic; maximum = delta[nx - 1][ic]; }
    for (let it = nx - 1; it >= 0; it--) {
      const c = p.frames[it].cand;
      const tmp = c[0]; c[0] = c[place]; c[place] = tmp;
      place = psi[it][place];
    }
  }

  // ---------- Pitch helpers ----------
  const pitchVoicedAt = (p, i) => isVoicedF(p.frames[i - 1].cand[0].f, p.ceiling); // i 1-based
  function pitchValueAtTime(p, x) {
    if (x < p.xmin || x > p.xmax) return NaN;
    const ireal = (x - p.x1) / p.dx + 1;
    const ileft = Math.floor(ireal);
    let phase = ireal - ileft, inear, ifar;
    if (phase < 0.5) { inear = ileft; ifar = ileft + 1; } else { ifar = ileft; inear = ileft + 1; phase = 1.0 - phase; }
    if (inear < 1 || inear > p.nx) return NaN;
    if (!pitchVoicedAt(p, inear)) return NaN;
    const fnear = p.frames[inear - 1].cand[0].f;
    if (ifar < 1 || ifar > p.nx) return fnear;
    if (!pitchVoicedAt(p, ifar)) return fnear;
    const ffar = p.frames[ifar - 1].cand[0].f;
    return fnear + phase * (ffar - fnear);
  }
  function voicedIntervalAfter(p, after) {
    let ileft = Math.ceil((after - p.x1) / p.dx) + 1;
    if (ileft > p.nx) return null;
    if (ileft < 1) ileft = 1;
    for (; ileft <= p.nx; ileft++) if (pitchVoicedAt(p, ileft)) break;
    if (ileft > p.nx) return null;
    let iright;
    for (iright = ileft; iright <= p.nx; iright++) if (!pitchVoicedAt(p, iright)) break;
    iright--;
    let tleft = p.x1 + (ileft - 1) * p.dx - 0.5 * p.dx;
    let tright = p.x1 + (iright - 1) * p.dx + 0.5 * p.dx;
    if (tleft >= p.xmax - 0.5 * p.dx) return null;
    if (tleft < p.xmin) tleft = p.xmin;
    if (tright > p.xmax) tright = p.xmax;
    if (tright <= after) return null;
    return [tleft, tright];
  }

  // ---------- PointProcess (periodic, cc) ----------
  function findExtremum3(z, d, n) {
    if (n < 3) {
      if (n <= 0) return 0;
      if (n === 1) return 1;
      const xl = Math.abs(z[d + 1]), xr = Math.abs(z[d + 2]);
      return xl > xr ? 1 : xl < xr ? 2 : 1.5;
    }
    let imin = 1, imax = 1, minimum = z[d + 1], maximum = z[d + 1];
    for (let i = 2; i <= n; i++) {
      const v = z[d + i];
      if (v < minimum) { minimum = v; imin = i; }
      if (v > maximum) { maximum = v; imax = i; }
    }
    if (minimum === maximum) return 0.5 * (n + 1.0);
    const iextr = Math.abs(minimum) > Math.abs(maximum) ? imin : imax;
    if (iextr === 1) return 1;
    if (iextr === n) return n;
    const vm = z[d + iextr], vl = z[d + iextr - 1], vr = z[d + iextr + 1];
    return iextr + (0.5 * (vr - vl)) / (2 * vm - vl - vr);
  }
  function findExtremum(s, tmin, tmax) {
    const imin = Math.max(1, xToLowIndex(s, tmin));
    const imax = Math.min(xToHighIndex(s, tmax), s.nx);
    const ie = findExtremum3(s.z, imin - 1, imax - imin + 1);
    return ie !== 0 ? s.x1 + (imin - 1 + ie - 1) * s.dx : 0.5 * (tmin + tmax);
  }
  function findMaximumCorrelation(s, t1, windowLength, tmin2, tmax2) {
    let maxCorr = -1.0, r1b = NaN, r3b = NaN, ir = NaN, r1 = 0, r2 = 0, r3 = 0, peak = 0, tout = NaN;
    const half = 0.5 * windowLength;
    const ileft1 = xToNearestIndex(s, t1 - half), iright1 = xToNearestIndex(s, t1 + half);
    const il2min = xToLowIndex(s, tmin2 - half), il2max = xToHighIndex(s, tmax2 - half);
    const z = s.z, nx = s.nx;
    for (let ileft2 = il2min; ileft2 <= il2max; ileft2++) {
      let n1 = 0, n2 = 0, prod = 0, lp = 0;
      for (let i1 = ileft1, i2 = ileft2; i1 <= iright1; i1++, i2++) {
        if (i1 < 1 || i1 > nx || i2 < 1 || i2 > nx) continue;
        const a1 = z[i1], a2 = z[i2];
        n1 += a1 * a1; n2 += a2 * a2; prod += a1 * a2;
        if (Math.abs(a2) > lp) lp = Math.abs(a2);
      }
      r1 = r2; r2 = r3;
      r3 = n1 === 0 || n2 === 0 ? 0 : prod / Math.sqrt(n1 * n2);
      if (r2 > maxCorr && r2 >= r1 && r2 >= r3) { r1b = r1; maxCorr = r2; r3b = r3; ir = ileft2 - 1; peak = lp; }
    }
    if (maxCorr > -1.0) {
      let interp = NaN;
      const d2r = (maxCorr - r1b) + (maxCorr - r3b);
      if (d2r !== 0) {
        const dr = 0.5 * (r3b - r1b);
        interp = maxCorr + (0.5 * dr * dr) / d2r;
        ir += dr / d2r;
      }
      const peakTime = t1 + (ir - ileft1) * s.dx;
      if (peakTime >= tmin2 && peakTime <= tmax2) {
        tout = peakTime;
        if (!Number.isNaN(interp)) maxCorr = interp;
      } else {
        const md = Math.sqrt((tmin2 - t1) * (tmax2 - t1));
        tout = tmin2 < t1 ? t1 - md : t1 + md;
        if (r3 > maxCorr) maxCorr = r3;
      }
    }
    return { corr: maxCorr, tout, peak };
  }
  function addPoint(arr, t) {
    // sisip terurut, tanpa duplikat
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; }
    if (arr[lo] === t) return;
    arr.splice(lo, 0, t);
  }
  function pointProcessCC(s, p) {
    const pts = [];
    let t = p.xmin, addedRight = -1e308;
    let globalPeak = 0;
    for (let i = 1; i <= s.nx; i++) { const v = Math.abs(s.z[i]); if (v > globalPeak) globalPeak = v; }
    let guard = 0;
    for (;;) {
      if (++guard > 100000) break;
      const iv = voicedIntervalAfter(p, t);
      if (!iv) break;
      const [tleft, tright] = iv;
      const tmiddle = 0.5 * (tleft + tright);
      const f0middle = pitchValueAtTime(p, tmiddle);
      if (Number.isNaN(f0middle)) { t = tright; continue; }
      let tmax = findExtremum(s, tmiddle - 0.5 / f0middle, tmiddle + 0.5 / f0middle);
      addPoint(pts, tmax);
      const tsave = tmax;
      for (let k = 0; k < 100000; k++) {
        const f0 = pitchValueAtTime(p, tmax);
        if (Number.isNaN(f0)) break;
        const res = findMaximumCorrelation(s, tmax, 1.0 / f0, tmax - 1.25 / f0, tmax - 0.8 / f0);
        if (!Number.isNaN(res.tout)) tmax = res.tout;
        if (res.corr === -1.0) tmax -= 1.0 / f0;
        if (tmax < tleft) {
          if (res.corr > 0.7 && res.peak > 0.023333 * globalPeak && tmax - addedRight > 0.8 / f0) addPoint(pts, tmax);
          break;
        }
        if (res.corr > 0.3 && (res.peak === 0 || res.peak > 0.01 * globalPeak)) {
          if (tmax - addedRight > 0.8 / f0) addPoint(pts, tmax);
        }
      }
      tmax = tsave;
      for (let k = 0; k < 100000; k++) {
        const f0 = pitchValueAtTime(p, tmax);
        if (Number.isNaN(f0)) break;
        const res = findMaximumCorrelation(s, tmax, 1.0 / f0, tmax + 0.8 / f0, tmax + 1.25 / f0);
        if (!Number.isNaN(res.tout)) tmax = res.tout;
        if (res.corr === -1.0) tmax += 1.0 / f0;
        if (tmax > tright) {
          if (res.corr > 0.7 && res.peak > 0.023333 * globalPeak) { addPoint(pts, tmax); addedRight = tmax; }
          break;
        }
        if (res.corr > 0.3 && (res.peak === 0 || res.peak > 0.01 * globalPeak)) { addPoint(pts, tmax); addedRight = tmax; }
      }
      t = tright;
    }
    return pts;
  }

  // ---------- jitter, shimmer ----------
  function isPeriod(t, ileft, pmin, pmax, maxFactor) {
    // t: array 0-based, ileft 1-based
    const nt = t.length, iright = ileft + 1;
    if (ileft < 1 || iright > nt) return false;
    if (pmin === pmax) return true;
    const interval = t[iright - 1] - t[ileft - 1];
    if (interval <= 0 || interval < pmin || interval > pmax) return false;
    if (!(maxFactor >= 1.0)) return true;
    const prev = ileft <= 1 ? NaN : t[ileft - 1] - t[ileft - 2];
    const next = iright >= nt ? NaN : t[iright] - t[iright - 1];
    let pf = !Number.isNaN(prev) && prev > 0 ? interval / prev : NaN;
    let nf = !Number.isNaN(next) && next > 0 ? interval / next : NaN;
    if (Number.isNaN(pf) && Number.isNaN(nf)) return true;
    if (!Number.isNaN(pf) && pf > 0 && pf < 1) pf = 1 / pf;
    if (!Number.isNaN(nf) && nf > 0 && nf < 1) nf = 1 / nf;
    if (!Number.isNaN(pf) && pf <= maxFactor) return true;
    if (!Number.isNaN(nf) && nf <= maxFactor) return true;
    return false;
  }
  function meanPeriod(t, pmin, pmax, maxFactor) {
    let n = t.length - 1;
    if (n < 1) return NaN;
    let sum = 0;
    for (let i = 1; i < t.length; i++) {
      if (isPeriod(t, i, pmin, pmax, maxFactor)) sum += t[i] - t[i - 1];
      else n--;
    }
    return n > 0 ? sum / n : NaN;
  }
  function jitterLocal(t, pmin, pmax, maxFactor) {
    let n = t.length - 1;
    if (n < 2) return NaN;
    let sum = 0;
    for (let i = 1; i < t.length - 1; i++) {
      const p1 = t[i] - t[i - 1], p2 = t[i + 1] - t[i];
      const f = p1 > p2 ? p1 / p2 : p2 / p1;
      if (pmin === pmax || (p1 >= pmin && p1 <= pmax && p2 >= pmin && p2 <= pmax && f <= maxFactor)) sum += Math.abs(p1 - p2);
      else n--;
    }
    if (n < 2) return NaN;
    return sum / (n - 1) / meanPeriod(t, pmin, pmax, maxFactor);
  }
  function hannWindowedRms(s, tmid, wl, wr) {
    const imin = Math.max(1, 1 + Math.ceil((tmid - wl - s.x1) / s.dx));
    const imax = Math.min(s.nx, 1 + Math.floor((tmid + wr - s.x1) / s.dx));
    if (imax - imin + 1 < 3) return NaN;
    let ss = 0, ws = 0;
    for (let i = imin; i <= imax; i++) {
      const tt = s.x1 + (i - 1) * s.dx;
      const width = tt < tmid ? wl : wr;
      const w = 0.5 + 0.5 * Math.cos((PI * (tt - tmid)) / width);
      const v = s.z[i] * w;
      ss += v * v; ws += w * w;
    }
    return Math.sqrt(ss / ws);
  }
  function shimmerLocal(t, s, pmin, pmax, maxFactor, maxAmpFactor) {
    if (t.length < 3) return NaN;
    const times = [], amps = [];
    for (let i = 1; i < t.length - 1; i++) {
      const p1 = t[i] - t[i - 1], p2 = t[i + 1] - t[i];
      const f = p1 > p2 ? p1 / p2 : p2 / p1;
      if (pmin === pmax || (p1 >= pmin && p1 <= pmax && p2 >= pmin && p2 <= pmax && f <= maxFactor)) {
        const pk = hannWindowedRms(s, t[i], 0.2 * p1, 0.2 * p2);
        if (!Number.isNaN(pk) && pk > 0) { times.push(t[i]); amps.push(pk); }
      }
    }
    let n = 0, num = 0;
    for (let i = 1; i < amps.length; i++) {
      const p = times[i] - times[i - 1];
      if (pmin === pmax || (p >= pmin && p <= pmax)) {
        const a1 = amps[i - 1], a2 = amps[i];
        const af = a1 > a2 ? a1 / a2 : a2 / a1;
        if (af <= maxAmpFactor) { num += Math.abs(a1 - a2); n++; }
      }
    }
    if (n < 1) return NaN;
    num /= n;
    let den = 0, m = 0;
    for (let i = 0; i < amps.length - 1; i++) { den += amps[i]; m++; }
    den /= m;
    if (!(den > 0)) return NaN;
    return num / den;
  }

  // ---------- fitur Praat lengkap (extract_praat_features di notebook) ----------
  function praatFeatures(y, sr, f0min, f0max) {
    f0min = f0min || 75; f0max = f0max || 500;
    const out = { f0_mean: NaN, f0_std: NaN, f0_min: NaN, f0_max: NaN, voiced_ratio: NaN,
      jitter_local_pct: NaN, shimmer_local_pct: NaN, hnr_mean: NaN };
    const snd = makeSound(y, sr);
    let pitch = null;
    try {
      pitch = toPitch(snd, { method: AC_HANNING, periodsPerWindow: 3.0, dt: 0.01, pitchFloor: f0min, pitchCeiling: f0max,
        maxnCandidates: 15, silenceThreshold: 0.03, voicingThreshold: 0.45, octaveCost: 0.01, octaveJumpCost: 0.35, voicedUnvoicedCost: 0.14 });
      const f0 = pitch.frames.map((fr) => fr.cand[0].f);
      const voiced = f0.filter((v) => v > 0);
      out.voiced_ratio = voiced.length / Math.max(f0.length, 1);
      if (voiced.length > 0) {
        let s = 0, lo = Infinity, hi = -Infinity;
        for (const v of voiced) { s += v; if (v < lo) lo = v; if (v > hi) hi = v; }
        const m = s / voiced.length;
        let vv = 0;
        for (const v of voiced) vv += (v - m) * (v - m);
        out.f0_mean = m; out.f0_std = Math.sqrt(vv / voiced.length); out.f0_min = lo; out.f0_max = hi;
      }
    } catch (e) { pitch = null; }
    try {
      if (pitch) {
        // "To PointProcess (periodic, cc)" memanggil Sound_to_Pitch(me, 0.0, fmin, fmax): time step 0.75/fmin = 0.01
        const pitchPP = 3.0 / f0min / 4.0 === 0.01 ? pitch : toPitch(snd, { method: AC_HANNING, periodsPerWindow: 3.0, dt: 0,
          pitchFloor: f0min, pitchCeiling: f0max, maxnCandidates: 15, silenceThreshold: 0.03, voicingThreshold: 0.45,
          octaveCost: 0.01, octaveJumpCost: 0.35, voicedUnvoicedCost: 0.14 });
        const pts = pointProcessCC(snd, pitchPP);
        out.jitter_local_pct = jitterLocal(pts, 0.0001, 0.02, 1.3) * 100;
        out.shimmer_local_pct = shimmerLocal(pts, snd, 0.0001, 0.02, 1.3, 1.6) * 100;
      }
    } catch (e) { /* NaN */ }
    try {
      const hp = toPitch(snd, { method: FCC_ACCURATE, periodsPerWindow: 1.0, dt: 0.01, pitchFloor: f0min, pitchCeiling: 0.5 / snd.dx,
        maxnCandidates: 15, silenceThreshold: 0.1, voicingThreshold: 0, octaveCost: 0, octaveJumpCost: 0, voicedUnvoicedCost: 0 });
      let sum = 0, n = 0;
      for (const fr of hp.frames) {
        const c = fr.cand[0];
        if (c.f === 0) continue;
        const r = c.s;
        const h = r <= 1e-15 ? -150 : r > 1 - 1e-15 ? 150 : 10 * Math.log10(r / (1 - r));
        sum += h; n++;
      }
      out.hnr_mean = n ? sum / n : NaN;
    } catch (e) { /* NaN */ }
    return out;
  }

  const Praat = { makeSound, toPitch, pointProcessCC, jitterLocal, shimmerLocal, praatFeatures, interpolateSinc };
  if (typeof module !== "undefined" && module.exports) module.exports = Praat;
  else root.Praat = Praat;
})(typeof self !== "undefined" ? self : this);
