# Tugas 2 NLP - Proposal Pipeline & Eksperimen Baseline

Prediksi arah kurs USD/IDR (JISDOR BI) dari berita geopolitik CNBC dengan NLP klasik
(VADER, Loughran-McDonald, TF-IDF/BoW; tanpa transformer atau embedding pre-trained).

## Formulasi singkat
- Target: `y_T = 1` jika `kurs_T > kurs_(T-1)` (IDR melemah), selain itu 0.
- Berita bertanggal d dipakai untuk hari kerja pertama **setelah** d (Jumat/Sabtu/Minggu -> Senin).
- Split kronologis 60/20/20: train 720, val 240, test 241 hari. Test dipakai sekali.
- Metrik: Directional Accuracy, F1 macro, MCC, AUC; uji binomial vs majority, bootstrap CI, McNemar.

## Cara menjalankan (dari folder `src/`)
```bash
pip install -r requirements.txt
python split.py            # target, penautan berita, split  -> data/processed/daily_base.csv, articles_targeted.csv
python preprocessing.py    # teks bersih                      -> data/processed/articles_preprocessed.csv
python features.py         # lexicon, agregasi, TF-IDF/BoW    -> data/processed/features_daily.csv
python run_experiments.py  # baseline + model gabungan        -> results/*.csv
python export_splits.py    # data/train.csv, val.csv, test.csv, data/splits/*_articles.csv
```
Input: `data/cleaned/aligned_news_kurs.csv` dan `data/cleaned/bi_jisdor_2021_2026.csv` (keluaran Tugas 1).
File antara berukuran besar di `data/processed/` (`articles_*.csv`, `article_features.csv`) tidak di-commit
(lihat `.gitignore`) dan dibuat ulang oleh skrip di atas.

## Struktur
```
data/train.csv val.csv test.csv     matriks fitur harian per split
data/splits/*_articles.csv          artikel per split
src/                                config, split, preprocessing, features, dataset,
                                    baseline, combined_model, evaluate, run_experiments, export_splits
notebook/01_eda.ipynb               EDA
notebook/02_baseline_experiments.ipynb   hasil eksperimen dan ablation
results/                            metrics.csv, predictions_test.csv, mcnemar_vs_kurs_only.csv
report/                             Laporan_Tugas2_NLP.pdf, pipeline.png/.dot, make_report.py
```

## Hasil awal (test, 241 hari; majority = 0,573)
Tidak ada model yang berbeda signifikan dari majority baseline, dan fitur NLP tidak meningkatkan model
kurs-saja (AUC sekitar 0,5). Detail dan keterbatasan ada di laporan dan notebook 02.
