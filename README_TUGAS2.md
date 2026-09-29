# Tugas 2 NLP - Proposal Pipeline & Eksperimen Baseline

Prediksi arah kurs USD/IDR (JISDOR BI) dari berita geopolitik CNBC dengan NLP klasik
(VADER, Loughran-McDonald, TF-IDF; tanpa transformer atau embedding pre-trained).

## Formulasi singkat
- Target: `y_T = 1` jika `kurs_T > kurs_(T-1)` (IDR melemah), selain itu 0.
- Berita bertanggal d dipakai untuk hari kerja pertama **setelah** d (Jumat/Sabtu/Minggu -> Senin).
- Split kronologis 60/20/20: train 720, val 240, test 241 hari. Test dipakai sekali.
- Metrik: Directional Accuracy dan F1 macro.

## Model
- Baseline: majority, persistence (naive), ARIMA(1,0,1), XGBoost dengan fitur kurs saja.
- Model gabungan: XGBoost dengan fitur kurs + lexicon / TF-IDF / keduanya.

## Cara menjalankan (dari folder `src/`)
```bash
pip install -r requirements.txt
python split.py            # target, penautan berita, split  -> data/processed/daily_base.csv
python preprocessing.py    # teks bersih
python features.py         # lexicon, agregasi harian, TF-IDF + SVD -> data/processed/features_daily.csv
python run_experiments.py  # baseline + model gabungan -> results/metrics.csv, predictions_test.csv
python export_splits.py    # data/train.csv, val.csv, test.csv
python ../report/make_report.py   # laporan PDF (butuh results/metrics.csv)
```
Input: `data/cleaned/aligned_news_kurs.csv` dan `data/cleaned/bi_jisdor_2021_2026.csv` (keluaran Tugas 1).
File antara besar di `data/processed/` (`articles_*.csv`, `article_features.csv`) tidak di-commit
(lihat `.gitignore`) dan dibuat ulang oleh skrip di atas.

## Struktur
```
data/train.csv val.csv test.csv     matriks fitur harian per split
data/splits/*_articles.csv          artikel per split
src/                                config, split, preprocessing, features, dataset,
                                    baseline, model, evaluate, run_experiments, export_splits
notebook/01_eda.ipynb               EDA
notebook/02_baseline_experiments.ipynb   hasil eksperimen
results/                            metrics.csv, predictions_test.csv (dibuat run_experiments.py)
report/                             make_report.py, pipeline.png/.dot, Laporan_Tugas2_NLP.pdf (dibuat make_report.py)
```
