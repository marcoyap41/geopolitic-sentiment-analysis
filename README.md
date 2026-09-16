# Tugas 1 NLP - KOM

**Dosen Pengampu:** Yunita Sari, Ph.D.

**Anggota Kelompok:**
1. DHIMAS PUTRA SULISTIO (24/537952/PA/22811)
2. BOBBY RAHMAN HARTANTO (24/539383/PA/22903)
3. MARCO CHRISTIAN (24/539714/PA/22915)
4. Raditya Nathaniel Nugroho (24/543188/PA/23069)

---

## Workflow Eksekusi Project (Data Pipeline)

Berikut adalah alur kerja (*workflow*) pemrosesan data berita geopolitik CNBC dan penyelarasan (*temporal alignment*) dengan data pergerakan nilai tukar (Kurs) JISDOR Bank Indonesia:

### Langkah 1: Ekstraksi URL Sitemap
* **Script Eksekusi:** `src/scraper_cnbc_sitemap.py`
* **Deskripsi:** Melakukan *fetching* pada `CNBC sitemapAll.xml`, memfilter rentang tanggal, dan mengekstrak URL serta tanggal terbit artikel.
* **Output CSV:** `data/raw/cnbc_sitemap_urlonly_*.csv`

### Langkah 2: Scraping Judul Asli (Parallel Chunking)
* **Script Eksekusi:** `src/fetch_real_titles.py`
* **Deskripsi:** Membaca data URL mentah, lalu membagi data menjadi 4 *chunk* (bagian) untuk mempercepat proses *scraping* judul asli secara simultan langsung dari HTML halaman web. Keempat hasil *chunk* kemudian digabungkan (*merge*).
* **Bukti/File Chunk:** `data/raw/cnbc_real_titles.chunk*of4.csv` (beserta file checkpoint-nya).
* **Output CSV:** `data/raw/cnbc_real_titles_merged_20260916_023442.csv`

### Langkah 3: Preprocessing Tahap 1 (Cleaning & Geopolitical Filter)
* **Script Eksekusi:** `src/preprocessing_pipeline.py`
* **Input Data:** `data/raw/cnbc_real_titles_merged_20260916_023442.csv`
* **Deskripsi:** 
  1. Deduplikasi berdasarkan URL.
  2. Pengecekan ulang dan filter rentang tanggal (1 Sep 2021 - 1 Sep 2026).
  3. Filter URL non-artikel.
  4. Pembersihan teks judul.
  5. Penghapusan baris dengan judul kosong.
  6. Filtering teks berdasarkan 8 kategori kata kunci Geopolitik & Makro Ekonomi.
* **Output CSV:** `data/cleaned/news_geopolitik_clean.csv`

### Langkah 4: Scraping Keypoints Artikel
* **Script Eksekusi:** `src/scraper_keypoints.py`
* **Input Data:** `data/cleaned/news_geopolitik_clean.csv`
* **Deskripsi:** Mengunjungi setiap URL berita yang sudah bersih untuk mengambil *Keypoints* (ringkasan 3-4 poin utama dari editor CNBC).
* **Output CSV:** `data/raw/cnbc_keypoints.csv`

### Langkah 5: Preprocessing Tahap 2 (Negative Filtering & Disambiguasi)
* **Script Eksekusi:** `src/filter_negative.py`
* **Input Data:** `data/cleaned/news_geopolitik_clean.csv` & `data/raw/cnbc_keypoints.csv`
* **Deskripsi:** Menerapkan strategi filter lanjutan:
  - Menggunakan *URL Whitelist/Blacklist* untuk keseluruhan data.
  - Menerapkan *Negative Words Filtering* (contoh: membuang entitas *Taylor Swift*, acara *Shark Tank*, dll) khusus pada artikel yang memiliki *keypoints* untuk menekan metrik *False Positive*.
* **Output CSV:** `data/cleaned/news_geopolitik_final.csv`

### Langkah 6: Pembersihan Data Kurs BI (Alur Paralel)
* **Script Eksekusi:** `src/process_bi_rate.py`
* **Input Data:** `Informasi Kurs Jisdor BI.xlsx`
* **Deskripsi:** Membaca data mentah dari website Bank Indonesia, menghitung nilai perubahan kurs harian (`kurs_change` & `kurs_change_pct`), serta membersihkan format tanggal menjadi struktur yang siap diolah.
* **Output CSV:** `data/cleaned/bi_jisdor_2021_2026.csv`

### Langkah 7: Temporal Alignment (Penggabungan Data Final)
* **Script Eksekusi:** `src/temporal_alignment.py`
* **Input Data:** 
  - `data/cleaned/news_geopolitik_final.csv`
  - `data/cleaned/bi_jisdor_2021_2026.csv`
* **Deskripsi:** Menyelaraskan (*align*) tanggal terbit artikel berita dengan kalender hari kerja (*trading days*) Bank Indonesia. Menggeser berita yang rilis pada akhir pekan atau hari libur nasional ke hari kerja berikutnya, sehingga kejadian geopolitik dipasangkan dengan pergerakan kurs yang relevan secara waktu nyata. Terakhir, mengekstrak arah pergerakan kurs (*kurs direction*).
* **Final Output CSV:** `data/cleaned/aligned_news_kurs.csv` (berisi metadata lengkap: tanggal, *aligned date*, URL, *title*, *keypoints*, *keyword category*, *kurs_idr_per_usd*, dan *kurs direction*).
