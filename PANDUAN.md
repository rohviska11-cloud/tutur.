# tutur. — panduan website (Random Forest, layout sidebar)

Website statis untuk GitHub Pages, tanpa server. Semua langkah (baca audio → 66 fitur → Random Forest)
berjalan di browser pengunjung, jadi tidak kena limit seperti Hugging Face / Streamlit.

| Skenario | Model | File di folder `model/` |
|---|---|---|
| 1 · Status bicara | Random Forest · MFCC-Statistik (52) | `speech_status.json`, `speech_status.bin` |
| 2 · Tingkat keparahan | Random Forest · MFCC + Akustik (66) | `severity.json`, `severity.bin` |

Pengaturan RF: `n_estimators=300, max_depth=None, min_samples_leaf=5, class_weight="balanced_subsample"`.

## 1. Ekspor model dari Colab

1. Buka notebook `TORGO_Final_2Skenario_RF_ECAPA.ipynb` (atau notebook Bab IX, karena variabelnya sama) dan jalankan sampai Bab VII.
2. Tambah sel baru di paling bawah, tempel isi **`colab/sel_ekspor_web.py`**, lalu jalankan.
3. Sel ini melatih ulang kedua RF dengan setelan terbaik, mengecek bahwa file ekspor identik dengan sklearn
   (harus muncul selisih sekitar 1e-7), lalu mengunduh `tutur_web_model.zip`.
4. Ekstrak zip. Isinya folder `model/` (4 file) dan `parity/` (untuk `cek.html`).

## 2. File yang diunggah ke GitHub

**Unggah / timpa:**

```
index.html
cek.html
.nojekyll
assets/css/style.css
assets/js/app.js
assets/js/worker.js
assets/js/forest.js
assets/js/dsp.js
assets/js/praat.js
assets/js/features.js
model/speech_status.json     (dari zip Colab)
model/speech_status.bin      (dari zip Colab)
model/severity.json          (dari zip Colab)
model/severity.bin           (dari zip Colab)
```

**Hapus dari repo bila sempat diunggah (versi WavLM/HuBERT, tidak dipakai lagi):**

```
assets/js/svm.js
model/manifest.json
model/svm_speech_status.json, model/svm_speech_status.bin
model/svm_severity.json, model/svm_severity.bin
model/wavlm_l8.onnx.part000 ... part009
model/hubert_l12.onnx.part000 ... part012
```

Semua file berukuran kecil (model RF sekitar 2 MB per skenario), jadi bisa diunggah lewat
**Add file → Upload files** di GitHub.

## 3. Coba di komputer sendiri (opsional)

Browser tidak menjalankan Web Worker dari `file://`, jadi pakai server kecil:

```bash
cd torgo-web
```

```bash
python -m http.server 8000
```

Buka `http://localhost:8000`. Tombol **Klasifikasikan** aktif setelah ada suara dan model berhasil dimuat.

`http://localhost:8000/cek.html` (perlu folder `parity/`) membandingkan fitur dan prediksi notebook dengan hasil browser:

- MFCC, ZCR, RMS, centroid: selisih relatif sangat kecil (di bawah 0,1%).
- F0, jitter, shimmer, HNR: kecil, bisa sedikit lebih besar karena merupakan port dari Praat.
- Kolom "status sama" dan "severity sama" sebaiknya (hampir) penuh.

Setelah dicek, folder `parity/` boleh dihapus.

## 4. Mengaktifkan GitHub Pages

Jika repo sudah aktif, cukup unggah file di atas dan tunggu 1–2 menit.
Untuk repo baru: **Settings → Pages → Deploy from a branch → `main` / `(root)`**.

## 5. Mengubah tampilan

- Warna: variabel di atas `assets/css/style.css` (`--deep`, `--teal`, `--slate`, `--paper`).
- Lebar sidebar: `--side` di `style.css`. Menu sidebar dan kartu "Model aktif": bagian `<aside class="side">` di `index.html`.
- Kalimat contoh untuk direkam: array `PROMPTS` di `assets/js/app.js`.
- Angka akurasi: bagian `#kinerja` di `index.html` (cari `data-val`).
- Tabel metode, Pojok Bacaan, dan tautan referensi: bagian `#metode` dan `#bacaan` di `index.html`.

## 6. Kalau ada masalah

| Gejala | Penyebab & solusi |
|---|---|
| Muncul "Model belum tersedia" | 4 file di `model/` belum diunggah atau namanya berbeda |
| Tombol Klasifikasikan tetap abu-abu | belum ada rekaman/file, atau model gagal dimuat (lihat poin di atas) |
| Halaman blank saat dibuka dengan klik dua kali | harus lewat server (langkah 3) atau GitHub Pages |
| Tombol mikrofon tidak merespons | izin mikrofon ditolak; buka ikon gembok di address bar, izinkan mikrofon |
| File tidak bisa dibaca | ubah ke WAV/MP3 dulu |
| Hasil rekaman terasa beda dengan file TORGO | wajar: mikrofon, ruangan, dan bahasa berbeda dari data latih |
