# =====================================================================================
# BAB VIII. EKSPOR MODEL UNTUK WEBSITE GITHUB PAGES (TANPA SERVER)
# Tempel sel ini di PALING BAWAH notebook TORGO_Final_2Skenario_RF_ECAPA.ipynb,
# lalu jalankan setelah semua sel sebelumnya selesai (butuh dev_df, df_s2, ds, dll).
#
# Hasil: /content/tutur_web_model.zip berisi
#   model/speech_status.json + .bin   -> RF | MFCC-Statistik (52)
#   model/severity.json + .bin        -> RF | MFCC+Akustik (66)
#   parity/parity.json + clip_*.wav   -> bahan uji kecocokan fitur Python vs browser (cek.html)
# =====================================================================================
import os, json, shutil, time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

WEB_DIR = "/content/tutur_web_model"
if os.path.exists(WEB_DIR):
    shutil.rmtree(WEB_DIR)
os.makedirs(f"{WEB_DIR}/model", exist_ok=True)
os.makedirs(f"{WEB_DIR}/parity", exist_ok=True)

RF_PARAMS_S1 = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5}   # metode terbaik Skenario 1
RF_PARAMS_S2 = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5}   # hiperparameter tersering Skenario 2

# ---------- 1. Latih ulang model final persis seperti Bab VII ----------
# Skenario 1: train+val (dev_df), fitur 52 MFCC
web_pipe1 = make_pipe(make_rf(RF_PARAMS_S1)).fit(dev_df[MFCC_COLS], y_dev)
# Skenario 2: seluruh speaker df_s2, fitur 66 (MFCC + akustik)
web_pipe2 = make_pipe(make_rf(RF_PARAMS_S2)).fit(df_s2[ALL_FEAT_COLS], y_s2)
print("Kelas S1:", list(web_pipe1.classes_), "| Kelas S2:", list(web_pipe2.classes_))


# ---------- 2. Ubah Random Forest jadi file biner kecil ----------
def export_forest(pipe, cols, name, info):
    imp, sc, rf = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["model"]
    assert len(imp.statistics_) == len(cols) and np.all(np.isfinite(imp.statistics_)), "imputer membuang kolom"
    K = len(rf.classes_)
    THR, LEFT, RIGHT, FEAT, VAL, roots = [], [], [], [], [], []
    node_off, leaf_off = 0, 0
    for est in rf.estimators_:
        t = est.tree_
        is_leaf = t.children_left == -1
        leaf_idx = np.cumsum(is_leaf) - 1 + leaf_off
        thr = t.threshold.astype(np.float64)
        t32 = thr.astype(np.float32)
        over = t32.astype(np.float64) > thr                       # sklearn: X(float32) <= threshold(float64)
        t32[over] = np.nextafter(t32[over], np.float32(-np.inf))  # pembulatan ke bawah agar perbandingan identik
        THR.append(np.where(is_leaf, 0, t32).astype("<f4"))
        LEFT.append(np.where(is_leaf, leaf_idx, t.children_left + node_off).astype("<i4"))
        RIGHT.append(np.where(is_leaf, -1, t.children_right + node_off).astype("<i4"))
        FEAT.append(np.where(is_leaf, -1, t.feature).astype("<i2"))
        v = t.value[:, 0, :][is_leaf].astype(np.float64)
        VAL.append((v / v.sum(axis=1, keepdims=True)).astype("<f4"))
        roots.append(node_off)
        node_off += t.node_count
        leaf_off += int(is_leaf.sum())
    thr, left, right, feat = map(np.concatenate, (THR, LEFT, RIGHT, FEAT))
    val = np.concatenate(VAL).reshape(-1)
    bin_name = f"{name}.bin"
    with open(f"{WEB_DIR}/model/{bin_name}", "wb") as f:
        for arr in (thr, left, right, val, feat):
            f.write(arr.tobytes())
    meta = {
        "task": name, "bin": bin_name, "classes": [str(c) for c in rf.classes_], "feature_cols": list(cols),
        "imputer_median": imp.statistics_.tolist(), "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "n_trees": len(rf.estimators_), "n_classes": K, "n_nodes": int(node_off), "n_leaves": int(leaf_off),
        "roots": roots, "dibuat": time.strftime("%Y-%m-%d %H:%M"), **info,
    }
    with open(f"{WEB_DIR}/model/{name}.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return {"thr": thr, "left": left, "right": right, "feat": feat, "val": val.reshape(-1, K), "meta": meta}


def js_like_proba(ex, X):
    """Menjalankan file ekspor dengan logika yang sama seperti forest.js, untuk memastikan hasil identik."""
    m = ex["meta"]
    X = np.asarray(X, dtype=np.float64).copy()
    med = np.array(m["imputer_median"])
    X = np.where(np.isfinite(X), X, med)
    Z = ((X - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])).astype(np.float32)
    out = np.zeros((len(Z), m["n_classes"]))
    for i, z in enumerate(Z):
        for r in m["roots"]:
            n = r
            while ex["feat"][n] >= 0:
                n = ex["left"][n] if z[ex["feat"][n]] <= ex["thr"][n] else ex["right"][n]
            out[i] += ex["val"][ex["left"][n]]
    return out / len(m["roots"])


ex1 = export_forest(web_pipe1, MFCC_COLS, "speech_status",
                    {"nama": "Random Forest | MFCC-Statistik (52)", "params": RF_PARAMS_S1,
                     "metrics": {"accuracy": 0.6937, "precision": 0.7640, "recall_dysarthria": 0.5075, "macro_f1": 0.6788}})
ex2 = export_forest(web_pipe2, ALL_FEAT_COLS, "severity",
                    {"nama": "Random Forest | MFCC+Akustik (66)", "params": RF_PARAMS_S2,
                     "metrics": {"accuracy": 0.4787, "macro_f1": 0.3124, "balanced_accuracy": 0.3220, "adjacent_accuracy": 0.7769}})

# Uji: ekspor == sklearn
chk1 = test_df[MFCC_COLS].sample(min(200, len(test_df)), random_state=SEED)
chk2 = df_s2[ALL_FEAT_COLS].sample(min(200, len(df_s2)), random_state=SEED)
d1 = np.abs(js_like_proba(ex1, chk1.values) - web_pipe1.predict_proba(chk1)).max()
d2 = np.abs(js_like_proba(ex2, chk2.values) - web_pipe2.predict_proba(chk2)).max()
print(f"Selisih maks probabilitas ekspor vs sklearn -> S1: {d1:.2e} | S2: {d2:.2e}")
assert d1 < 1e-4 and d2 < 1e-4, "ekspor tidak identik dengan sklearn"


# ---------- 3. Bahan uji kecocokan fitur (dibuka lewat cek.html) ----------
def feats_from_array(y16):
    yp, sr = preprocess_audio(y16, TARGET_SR)
    f = mfcc_statistics(extract_mfcc_sequence(yp, sr))
    f.update(extract_librosa_features(yp, sr))
    f.update(extract_praat_features(yp, sr))
    return f

pick = pd.concat([
    test_df.groupby("y", group_keys=False).apply(lambda g: g.sample(min(5, len(g)), random_state=SEED)),
    df_s2.groupby("y_sev", group_keys=False).apply(lambda g: g.sample(min(3, len(g)), random_state=SEED)),
]).drop_duplicates("row_idx")

parity = []
for k, (_, row) in enumerate(pick.iterrows()):
    y, sr = decode_audio_entry(ds[int(row["row_idx"])]["audio"])
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).mean(axis=1))
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    fn = f"clip_{k:02d}.wav"
    sf.write(f"{WEB_DIR}/parity/{fn}", y, TARGET_SR, subtype="FLOAT")
    y_back, _ = sf.read(f"{WEB_DIR}/parity/{fn}", dtype="float32")
    f = feats_from_array(y_back)
    xf = pd.DataFrame([f])
    parity.append({
        "file": fn, "speaker": str(row["speaker_id"]), "label_status": str(row["y"]),
        "label_severity": str(row.get("severity", "")),
        "feats": {c: (None if not np.isfinite(f[c]) else float(f[c])) for c in ALL_FEAT_COLS},
        "py_status": web_pipe1.predict_proba(xf[MFCC_COLS])[0].tolist(),
        "py_severity": web_pipe2.predict_proba(xf[ALL_FEAT_COLS])[0].tolist(),
    })
with open(f"{WEB_DIR}/parity/parity.json", "w") as f:
    json.dump({"classes_status": list(map(str, web_pipe1.classes_)), "classes_severity": list(map(str, web_pipe2.classes_)),
               "clips": parity}, f)
print(f"Bahan uji: {len(parity)} klip")

# ---------- 4. Ringkasan ukuran & unduh ----------
for sub in ["model", "parity"]:
    tot = sum(os.path.getsize(os.path.join(WEB_DIR, sub, x)) for x in os.listdir(f"{WEB_DIR}/{sub}"))
    print(f"  {sub:7s}: {tot / 1e6:6.2f} MB")
shutil.make_archive("/content/tutur_web_model", "zip", WEB_DIR)
try:
    from google.colab import files
    files.download("/content/tutur_web_model.zip")
except Exception:
    print("Unduh manual: /content/tutur_web_model.zip")
