# =====================================================================================
# BAB X. EKSPOR MODEL TERBAIK (WavLM / HuBERT + SVM-RBF) UNTUK WEBSITE GITHUB PAGES
# Tempel sel ini di PALING BAWAH notebook "TORGO_DATASET_dengan_Bab_IX", lalu jalankan
# setelah sel Bab IX selesai (butuh df_x, split1, y1, X2, y2, ssl_cols, preprocess_audio, ds, dll).
#
#   Skenario 1: SVM-RBF | WavLM-Base+  lapisan 8,  C=1.0
#   Skenario 2: SVM-RBF (OvR) | HuBERT-Base lapisan 12, C=1.0
#
# Hasil: /content/tutur_web_model_ssl.zip  -> isi folder "model/" di repo GitHub Pages.
# File ONNX dipecah <= 19 MB per bagian, supaya bisa diunggah lewat tombol "Upload files" GitHub.
# =====================================================================================
!pip -q install onnx onnxruntime

import os, json, shutil, time, gc
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoFeatureExtractor, AutoModel

OUT = "/content/tutur_web_model_ssl"
MODEL_OUT = f"{OUT}/model"
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(f"{OUT}/parity", exist_ok=True)

BEST = {
    "speech_status": {"tag": "wavlm", "layer": 8, "C": 1.0, "ovr": False},
    "severity": {"tag": "hubert", "layer": 12, "C": 1.0, "ovr": True},
}
CHUNK_MB = 19
QUANTIZE = True          # int8 (MatMul) -> ukuran kira-kira 1/3. Set False bila agreement di bawah terlalu rendah.
SV_DTYPE = "float16"     # penyimpanan support vector; otomatis jatuh ke float32 bila tidak identik


# ---------------------------------------------------------------- 1. Ekspor ONNX
class SSLPool(torch.nn.Module):
    """Gelombang 16 kHz (sudah preprocess_audio) -> [mean_t, std_t] dari hidden_states[layer] = vektor 1536."""
    def __init__(self, hf_name, layer):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(hf_name)
        self.do_norm = bool(getattr(fe, "do_normalize", False))
        m = AutoModel.from_pretrained(hf_name).eval()
        m.config.layerdrop = 0.0
        m.encoder.layers = m.encoder.layers[:layer]        # lapisan setelah 'layer' tidak dibutuhkan
        self.m, self.layer = m, layer

    def forward(self, wave):                                 # wave: (1, n)
        if self.do_norm:                                     # sama dengan Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
            wave = (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        h = self.m(wave, output_hidden_states=True).hidden_states[self.layer][0]
        return torch.cat([h.mean(dim=0), h.std(dim=0)])


def ref_embed(hf_name, layer, y16):
    """Embedding persis seperti sel 9.2 (model penuh, presisi penuh)."""
    fe = AutoFeatureExtractor.from_pretrained(hf_name)
    mdl = AutoModel.from_pretrained(hf_name).eval()
    with torch.no_grad():
        inp = fe(y16, sampling_rate=16000, return_tensors="pt")["input_values"]
        h = mdl(inp, output_hidden_states=True).hidden_states[layer][0]
        return torch.cat([h.mean(dim=0), h.std(dim=0)]).numpy()


def prep(y, sr):
    y, sr = preprocess_audio(y, sr)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)))
    return y.astype(np.float32)


# klip uji nyata untuk verifikasi
te_rows = df_x.loc[split1 == "test", "row_idx"].astype(int).values
rng = np.random.default_rng(SEED)
probe_ids = rng.choice(te_rows, size=min(4, len(te_rows)), replace=False)
probes = [prep(*decode_audio_entry(ds[int(i)]["audio"])) for i in probe_ids]

onnx_files = {}
for task, b in BEST.items():
    tag, L = b["tag"], b["layer"]
    hf = SSL_MODELS[tag]
    net = SSLPool(hf, L)
    fp32 = f"/content/{tag}_l{L}_fp32.onnx"
    dummy = torch.from_numpy(probes[0])[None, :]
    torch.onnx.export(net, (dummy,), fp32, input_names=["wave"], output_names=["emb"],
                      dynamic_axes={"wave": {1: "n"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    final = fp32
    if QUANTIZE:
        final = f"/content/{tag}_l{L}_int8.onnx"
        quantize_dynamic(fp32, final, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    # verifikasi: ONNX (panjang berbeda-beda) vs PyTorch model penuh
    sess = ort.InferenceSession(final, providers=["CPUExecutionProvider"])
    for y in probes:
        e_ref = ref_embed(hf, L, y)
        e_onx = sess.run(None, {"wave": y[None, :]})[0]
        cos = float(np.dot(e_ref, e_onx) / (np.linalg.norm(e_ref) * np.linalg.norm(e_onx)))
        print(f"  {tag} L{L} | durasi {len(y)/16000:.2f}s | cosine ONNX vs PyTorch = {cos:.5f}")
    onnx_files[task] = final
    del net, sess
    gc.collect()


def split_file(path, base):
    data = open(path, "rb").read()
    step = CHUNK_MB * 1024 * 1024
    parts = []
    for k in range(0, len(data), step):
        name = f"{base}.part{k // step:03d}"
        with open(f"{MODEL_OUT}/{name}", "wb") as f:
            f.write(data[k:k + step])
        parts.append(name)
    return parts, len(data)


# ---------------------------------------------------------------- 2. Latih & ekspor SVM
def make_svm(C, ovr):
    svc = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("model", OneVsRestClassifier(svc) if ovr else svc)])


def platt(svc, Z):
    """Cari rumus sigmoid yang sama persis dengan predict_proba sklearn untuk SVC biner."""
    d = svc.decision_function(Z)
    p1 = svc.predict_proba(Z)[:, 1]
    A, B = float(svc.probA_[0]), float(svc.probB_[0])
    for s in (1.0, -1.0):
        for flip in (False, True):
            q = 1.0 / (1.0 + np.exp(A * (s * d) + B))
            q = 1 - q if flip else q
            if np.abs(q - p1).max() < 1e-6:
                return {"A": A, "B": B, "s": s, "flip": flip}
    raise RuntimeError("rumus Platt tidak cocok")


def export_svm(task, pipe, cols, X_train, info):
    imp, sc, mdl = pipe.named_steps["imputer"], pipe.named_steps["scaler"], pipe.named_steps["model"]
    Z = sc.transform(imp.transform(X_train))
    ests = mdl.estimators_ if isinstance(mdl, OneVsRestClassifier) else [mdl]
    union = sorted(set(np.concatenate([e.support_ for e in ests]).tolist()))
    pos = {v: i for i, v in enumerate(union)}
    rows = Z[union].astype(np.float32)
    E = []
    for e in ests:
        E.append({"idx": [pos[i] for i in e.support_.tolist()], "coef": e.dual_coef_[0].astype(float).tolist(),
                  "b": float(e.intercept_[0]), "gamma": float(e._gamma), "classes": [str(c) for c in e.classes_],
                  "platt": platt(e, Z[:300])})

    def js_decision(R, z):
        out = []
        for e in E:
            sv = R[e["idx"]]
            k = np.exp(-e["gamma"] * ((sv - z) ** 2).sum(1))
            out.append(float(np.dot(e["coef"], k) + e["b"]))
        return np.array(out)

    # pilih dtype penyimpanan + verifikasi decision_function
    chk = Z[np.random.default_rng(SEED).choice(len(Z), size=min(300, len(Z)), replace=False)]
    ref = np.stack([e.decision_function(chk) for e in ests], 1)
    dtype = SV_DTYPE
    R = rows.astype(np.float16).astype(np.float32) if dtype == "float16" else rows
    mine = np.stack([js_decision(R, z) for z in chk])
    same = (np.argmax(mine, 1) == np.argmax(ref, 1)).mean() if len(ests) > 1 else ((mine[:, 0] > 0) == (ref[:, 0] > 0)).mean()
    if same < 0.999 or np.abs(mine - ref).max() > 5e-3:
        dtype, R = "float32", rows
        mine = np.stack([js_decision(R, z) for z in chk])
    assert np.abs(mine - ref).max() < 5e-3, "rumus decision_function di website tidak cocok dengan sklearn"
    print(f"  {task}: {len(union)} support vector unik | simpan {dtype} | selisih decision maks {np.abs(mine - ref).max():.2e}")
    (rows.astype(np.float16) if dtype == "float16" else rows).tofile(f"{MODEL_OUT}/svm_{task}.bin")
    meta = {"task": task, "classes": [str(c) for c in mdl.classes_], "ovr": isinstance(mdl, OneVsRestClassifier),
            "n_feat": rows.shape[1], "n_rows": rows.shape[0], "dtype": dtype, "bin": f"svm_{task}.bin",
            "imputer_median": imp.statistics_.tolist(), "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "estimators": E, **info}
    with open(f"{MODEL_OUT}/svm_{task}.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    return meta


manifest = {"versi": time.strftime("%Y%m%d%H%M"), "tasks": {}}
svm_pipes = {}
for task, b in BEST.items():
    tag, L = b["tag"], b["layer"]
    cols = ssl_cols(tag, L)
    if task == "speech_status":
        dev = np.isin(split1, ["train", "val"])
        Xtr, ytr = X1.loc[dev, cols], y1[dev]
    else:
        Xtr, ytr = X2[cols], y2
    pipe = make_svm(b["C"], b["ovr"]).fit(Xtr, ytr)
    svm_pipes[task] = (pipe, cols)
    info = {"embed": SSL_NAMES[tag], "hf": SSL_MODELS[tag], "layer": L, "C": b["C"]}
    export_svm(task, pipe, cols, Xtr, info)
    parts, nbytes = split_file(onnx_files[task], f"{tag}_l{L}.onnx")
    manifest["tasks"][task] = {"onnx_parts": parts, "onnx_bytes": nbytes, "svm": f"svm_{task}.json", **info}

with open(f"{MODEL_OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)


# ---------------------------------------------------------------- 3. Agreement: ONNX di browser vs embedding notebook
N_CHECK = 120
sess = {t: ort.InferenceSession(onnx_files[t], providers=["CPUExecutionProvider"]) for t in BEST}
test_idx = np.where(split1 == "test")[0]
pick = np.random.default_rng(SEED).choice(test_idx, size=min(N_CHECK, len(test_idx)), replace=False)
agree = {t: [] for t in BEST}
parity = []
for n, i in enumerate(pick):
    rid = int(X1.iloc[i]["row_idx"])
    y = prep(*decode_audio_entry(ds[rid]["audio"]))
    rec = {}
    for task in BEST:
        pipe, cols = svm_pipes[task]
        e_onx = sess[task].run(None, {"wave": y[None, :]})[0][None, :]
        p_nb = pipe.predict(X1.iloc[[i]][cols])[0]                  # embedding cache notebook (GPU fp16)
        p_web = pipe.predict(pd.DataFrame(e_onx, columns=cols))[0]  # embedding ONNX (yang dipakai website)
        agree[task].append(p_nb == p_web)
        rec[task] = {"label": str(p_web), "proba": pipe.predict_proba(pd.DataFrame(e_onx, columns=cols))[0].tolist()}
    if n < 12:                                                      # bahan cek.html
        fn = f"clip_{n:02d}.wav"
        sf.write(f"{OUT}/parity/{fn}", y, 16000, subtype="FLOAT")
        parity.append({"file": fn, "label_status": str(y1[i]), "py": rec})
for t in BEST:
    print(f"Prediksi sama (embedding ONNX web vs embedding notebook) {t}: {np.mean(agree[t]):.1%} dari {len(agree[t])} klip")
with open(f"{OUT}/parity/parity.json", "w") as f:
    json.dump({"clips": parity}, f)

# ---------------------------------------------------------------- 4. Ringkasan & unduh
tot = 0
for fn in sorted(os.listdir(MODEL_OUT)):
    s = os.path.getsize(f"{MODEL_OUT}/{fn}")
    tot += s
    print(f"  model/{fn:34s} {s / 1e6:7.2f} MB")
print(f"Total folder model: {tot / 1e6:.1f} MB")
shutil.make_archive(OUT, "zip", OUT)
try:
    from google.colab import files
    files.download(OUT + ".zip")
except Exception:
    print("Unduh manual:", OUT + ".zip")
