# tutur. — panduan membuat & menayangkan website

Website statis untuk GitHub Pages. Tidak ada server, tidak ada Python di sisi web.
Semua langkah (baca audio → 66 fitur → Random Forest) berjalan di browser pengunjung,
jadi tidak kena limit seperti Hugging Face / Streamlit.

```
torgo-web/
├── index.html              halaman utama
├── cek.html                halaman uji kecocokan Python vs browser (opsional)
├── .nojekyll               supaya GitHub Pages tidak memproses file
├── assets/css/style.css    tampilan (deep teal, teal, slate)
├── assets/js/
│   ├── app.js              rekam, unggah, tampilan hasil
│   ├── worker.js           menghitung di thread terpisah
│   ├── dsp.js              replika librosa: preprocessing, MFCC, ZCR, RMS, centroid
│   ├── praat.js            replika Praat/parselmouth: F0, jitter, shimmer, HNR
│   ├── features.js         menyusun 66 fitur sesuai urutan kolom notebook
│   └── forest.js           menjalankan imputer → scaler → Random Forest
├── model/                  DIISI dari Colab (speech_status.*, severity.*)
└── colab/sel_ekspor_web.py sel yang ditempel ke notebook
```

## 1. Ekspor model dari Colab

1. Buka `TORGO_Final_2Skenario_RF_ECAPA.ipynb` dan jalankan semua sel sampai Bab VII.
2. Tambah satu sel kode di paling bawah, tempel isi `colab/sel_ekspor_web.py`, lalu jalankan.
3. Sel ini akan:
   - melatih ulang dua model final dengan setelan terbaik:
     - Skenario 1: RF | MFCC-Statistik (52), `n_estimators=300, max_depth=None, min_samples_leaf=5`, data train+val
     - Skenario 2: RF | MFCC+Akustik (66), setelan yang sama, seluruh speaker
   - mengubah tiap model jadi `*.json` (kelas, median imputer, mean/scale scaler) + `*.bin` (pohon)
   - mengecek bahwa file ekspor memberi probabilitas yang **sama persis** dengan sklearn (harus muncul selisih ~1e-7)
   - menyimpan ±20 klip uji ke `parity/`
   - mengunduh `tutur_web_model.zip`
4. Ekstrak zip. Salin isi folder `model/` ke `torgo-web/model/`. Salin juga folder `parity/` kalau mau menjalankan `cek.html`.

Ukuran model biasanya beberapa MB per skenario (dicetak di akhir sel). Masih jauh di bawah batas GitHub (100 MB per file).

## 2. Coba di komputer sendiri

Browser tidak mau menjalankan Web Worker dari `file://`, jadi pakai server kecil:

```bash
cd torgo-web
```

```bash
python -m http.server 8000
```

Buka `http://localhost:8000`. Tombol **Klasifikasikan** aktif setelah file dipilih jika model berhasil dimuat. Jika model tidak ditemukan, muncul pemberitahuan di bawah layar.

Lalu buka `http://localhost:8000/cek.html` dan tekan **Jalankan pengecekan**. Halaman ini membandingkan
fitur & prediksi notebook dengan hasil browser untuk klip yang sama. Yang diharapkan:

- MFCC, ZCR, RMS, centroid: selisih relatif sangat kecil (di bawah 0,1%).
- F0, jitter, shimmer, HNR: kecil, tapi bisa sedikit lebih besar karena ini port dari Praat.
- Kolom "status sama" dan "severity sama" sebaiknya (hampir) penuh.

Hasil cek ini bagus dilampirkan di laporan sebagai bukti website memakai model yang sama dengan notebook.
Setelah selesai, folder `parity/` boleh dihapus supaya repo tetap ringan.

## 3. Tayangkan di GitHub Pages

1. Buat repo baru di GitHub, misalnya `tutur` (Public).
2. Unggah semua isi `torgo-web/` (termasuk `.nojekyll` dan folder `model/`). Bisa lewat tombol
   **Add file → Upload files**, atau lewat git:

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "website tutur"
```

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/USERNAME/tutur.git
```

```bash
git push -u origin main
```

3. Di repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch**, pilih `main` dan `/ (root)`, simpan.
4. Tunggu 1–2 menit. Alamatnya `https://USERNAME.github.io/tutur/`.

Mikrofon hanya jalan di HTTPS atau localhost. github.io sudah HTTPS, jadi aman.

## 4. Cara kerja (versi singkat untuk laporan)

| Tahap | Di notebook | Di website |
|---|---|---|
| Baca audio | soundfile / librosa | `decodeAudioData` (wav, mp3, m4a, ogg, flac, rekaman mic) |
| Preprocessing | mono, resample 16 kHz, hapus DC, puncak 0,95, tanpa trimming | sama (`dsp.js`) |
| MFCC 52 | `librosa.feature.mfcc` 13 koef, n_fft 512, win 400, hop 160 | replika librosa: hann, 128 mel slaney, power_to_db top_db 80, DCT-II ortho |
| Akustik 14 | parselmouth + librosa | `praat.js` (port kode sumber Praat) + `dsp.js` |
| Model | `Pipeline(SimpleImputer median → StandardScaler → RandomForest)` | sama (`forest.js`), termasuk pembandingan float32 seperti sklearn |

Khusus rekaman mikrofon: hening panjang di awal dan akhir dipotong dengan sisa 0,3 detik,
karena orang biasanya diam sebentar setelah menekan tombol. File yang diunggah tidak dipotong sama sekali
(persis seperti notebook).

## 5. Mengubah tampilan

- Warna: variabel di atas `assets/css/style.css` (`--deep`, `--teal`, `--slate`, `--paper`).
- Font: Fraunces (judul) + Plus Jakarta Sans (isi), dari Google Fonts di `index.html`.
- Kalimat contoh untuk direkam: array `PROMPTS` di `assets/js/app.js`.
- Angka akurasi di bagian "Kinerja model": langsung di `index.html` (cari `data-val`).
- Tabel metode, isi Pojok Bacaan, dan tautan referensi: bagian `#metode` dan `#bacaan` di `index.html`.
- Nama label severity Indonesia: objek `SEV_ID` di `app.js`. Kelas yang digabung di notebook (misal `moderate+moderate-severe`) otomatis tampil sebagai "Sedang / Sedang-berat".

## 6. Kalau ada masalah

| Gejala | Penyebab & solusi |
|---|---|
| Muncul "Model belum tersedia" | file `model/speech_status.json`, `.bin`, `severity.json`, `.bin` belum diunggah, atau nama foldernya beda |
| Halaman blank saat dibuka dengan klik dua kali | harus lewat server (langkah 2) atau GitHub Pages |
| Tombol mikrofon tidak merespons | izin mikrofon ditolak; buka ikon gembok di address bar, izinkan mikrofon |
| File tidak bisa dibaca | format jarang (mis. AMR); ubah ke WAV/MP3 dulu |
| Hasil rekaman terasa beda dengan file TORGO | wajar: mikrofon, ruangan, dan bahasa berbeda dari data latih. Ucapkan kata bahasa Inggris, ruangan tenang |

## 7. Catatan jujur

- Model dilatih pada 15 penutur TORGO berbahasa Inggris. Suara baru (mic laptop, bahasa Indonesia, ruangan ramai)
  berada di luar data latih, jadi hasilnya perlu dibaca sebagai indikasi, bukan diagnosis.
- Skenario 2 (severity) akurasinya 47,9% (tebak acak 25%), dan 77,7% prediksinya meleset paling jauh satu tingkat.
  Ini sudah ditulis apa adanya di website.
