"""Membangun laporan PDF Tugas 2 dari results/metrics.csv dan report/pipeline.png."""
import ast
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
m = pd.read_csv(ROOT / "results/metrics.csv")
val, test = m[m.split == "val"].set_index("model"), m[m.split == "test"].set_index("model")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceBefore=12, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=4, keepWithNext=1)
T = ParagraphStyle("T", parent=B, fontSize=8.2, leading=10, alignment=0, spaceAfter=0)
TH = ParagraphStyle("TH", parent=T, fontName="Helvetica-Bold", textColor=colors.white)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), alignment=0)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=14, bulletIndent=4, spaceAfter=2)

def p(t): return Paragraph(t, B)
def bl(items): return [Paragraph(t, BUL, bulletText="\u2022") for t in items]

def tbl(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), TH if (header and i == 0) else T) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0c6")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5f7")))
    t.setStyle(TableStyle(st)); return t

def f(x, d=3): return "-" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")

story = []
story += [Paragraph("Proposal Pipeline &amp; Eksperimen Baseline", ParagraphStyle("t", parent=H1, fontSize=19, spaceAfter=2)),
          Paragraph("Prediksi Arah Nilai Tukar USD/IDR dari Berita Geopolitik CNBC dengan NLP Klasik", ParagraphStyle("st", parent=B, fontSize=11.5, leading=15, alignment=0)),
          Paragraph("Proyek NLP - Tugas 2 &nbsp;|&nbsp; Kelompok: [isi nama kelompok] &nbsp;|&nbsp; Anggota: [isi nama dan NIM anggota]", CAP),
          Spacer(1, 6)]

# ---------------- 1
story += [Paragraph("1. Formulasi Tugas", H1)]
story += [p("Hipotesis proyek adalah bahwa berita geopolitik global membantu memprediksi pergerakan kurs USD. Pada Tugas 2 hipotesis ini dioperasionalkan menjadi tugas klasifikasi biner harian dan diuji dengan pipeline NLP klasik (lexicon dan vektorisasi TF-IDF/BoW) tanpa model pre-trained atau transformer.")]
story += [Paragraph("1.1 Data", H2),
          p("Data berasal dari Tugas 1: 41.406 artikel CNBC (1 Sep 2021 - 1 Sep 2026) yang sudah difilter ke topik geopolitik dan makro (delapan kategori kata kunci), berisi judul dan, untuk 51,6% artikel, keypoints. Kurs yang dipakai adalah JISDOR Bank Indonesia harian. Setelah penautan tanggal diperoleh 1.201 hari kerja target, dengan rata-rata 34 artikel per hari.")]
story += [Paragraph("1.2 Definisi target", H2),
          p("Unit prediksi adalah hari kerja <b>T</b>. Target <b>y<sub>T</sub> = 1</b> jika kurs<sub>T</sub> &gt; kurs<sub>T-1</sub> (rupiah melemah terhadap USD), dan <b>y<sub>T</sub> = 0</b> jika sebaliknya. Terdapat 13 hari dengan kurs tidak berubah (1,1%) yang digabungkan ke kelas 0 agar deret hari tetap utuh dan fitur lag tidak terputus. Proporsi kelas 1 adalah 55,4% untuk seluruh data (train 56,1%, validasi 51,2%, test 57,3%).")]
story += [p("Klasifikasi biner dipilih dibanding regresi level kurs karena (i) kurs bersifat non-stasioner (14.284 menjadi 17.727 IDR/USD) sehingga level mudah diprediksi lewat tren tanpa ada hubungan dengan berita, (ii) hipotesis proyek berbicara tentang arah pergerakan, dan (iii) metrik arah mudah ditafsirkan dan dibandingkan dengan baseline. Regresi return dan target volatilitas dicadangkan untuk Tugas 3.")]
story += [Paragraph("1.3 Penautan berita ke target dan pencegahan data leakage", H2),
          p("Data tanggal CNBC hanya berisi tanggal (tanpa jam) dan mengikuti waktu AS, sehingga berita bertanggal d baru tercermin di pasar Asia pada hari kerja berikutnya. Aturan yang dipakai: <b>berita bertanggal d dipasangkan ke T, yaitu hari kerja pertama yang benar-benar setelah d</b>. Akibatnya berita Jumat, Sabtu, dan Minggu semuanya dipakai untuk memprediksi Senin, dan berita pada hari libur nasional dipakai untuk hari kerja berikutnya. Label kurs hari yang sama dengan berita (seperti kolom <i>kurs_direction</i> pada keluaran Tugas 1) tidak dipakai karena berita dan kurs pada tanggal yang sama dapat saling tumpang tindih waktu.")]
story += bl(["<b>Fitur kurs</b> untuk hari T hanya memakai data sampai T-1 (semua lag dibuat dengan shift).",
             "<b>Vocabulary, IDF, dan SVD</b> dari TF-IDF/BoW di-fit hanya pada hari train, lalu diterapkan ke validasi dan test.",
             "<b>Split kronologis tanpa shuffle</b>; hyperparameter dipilih dengan TimeSeriesSplit di dalam train; data test dipakai satu kali."])
story += [Paragraph("1.4 Pembagian data", H2)]
story += [tbl([["Split", "Periode target", "Jumlah hari", "P(y=1)"],
               ["Train (60%)", "2 Sep 2021 - 22 Agu 2024", "720", "0,561"],
               ["Validasi (20%)", "23 Agu 2024 - 28 Agu 2025", "240", "0,512"],
               ["Test (20%)", "29 Agu 2025 - 1 Sep 2026", "241", "0,573"]], [3.6*cm, 6*cm, 3*cm, 2.6*cm]), Spacer(1, 4),
          p("Untuk skor validasi, model dilatih pada train saja. Untuk skor test, model dilatih ulang pada train+validasi dengan hyperparameter yang dipilih lewat CV di train. Berkas hasil pembagian ada di <i>data/train.csv, val.csv, test.csv</i> (matriks fitur harian) dan <i>data/splits/*_articles.csv</i> (level artikel).")]
story += [Paragraph("1.5 Metrik evaluasi", H2)]
story += bl(["<b>Directional Accuracy (DA)</b>: proporsi arah yang benar; dibandingkan dengan <b>majority baseline</b> per split (kelas mayoritas train) dan <b>persistence</b> (arah hari T sama dengan T-1).",
             "<b>F1 macro</b> dan <b>Matthews Correlation Coefficient (MCC)</b>: mengungkap model yang hanya menebak kelas mayoritas (DA tinggi tetapi F1 macro rendah dan MCC mendekati nol). <b>AUC</b> dilaporkan untuk model berprobabilitas.",
             "<b>Uji statistik</b>: uji binomial satu sisi terhadap majority baseline, CI 95% bootstrap untuk DA, dan uji McNemar untuk membandingkan model gabungan dengan model kurs-saja. Dengan 241 hari test, satu standar error DA sekitar 3 poin, sehingga selisih di bawah sekitar 6 poin tidak dapat dibedakan dari kebetulan."])

# ---------------- 2
story += [Paragraph("2. Alasan Pemilihan Pendekatan Ekstraksi Fitur NLP", H1)]
story += [p("Ketentuan tugas melarang embedding pre-trained dan model berbasis transformer, sehingga seluruh fitur teks dibangun dari lexicon dan vektorisasi klasik. Dua keluarga fitur dipilih karena saling melengkapi: lexicon memberi sinyal <i>bermakna secara semantik</i> berdimensi rendah, sedangkan TF-IDF/BoW menangkap <i>topik dan kosakata</i> tanpa asumsi kamus.")]
story += [Paragraph("2.1 Unit teks: judul vs judul + keypoints", H2),
          p("Judul tersedia untuk 100% artikel, singkat, dan ditulis editor untuk merangkum kejadian, sehingga menjadi basis utama. Keypoints (ringkasan 3-4 poin dari editor CNBC) memberi konteks lebih kaya tetapi hanya ada pada 51,6% artikel dan cakupannya berubah antar tahun (sekitar 46% sampai 62%), yang dapat menimbulkan pergeseran distribusi fitur. Karena itu keduanya dibandingkan sebagai varian: <i>title</i> dan <i>title+keypoints</i> (memakai judul saja jika keypoints kosong). Isi artikel penuh tidak dipakai karena tidak tersedia dalam dataset dan biaya scraping ulang tinggi.")]
story += [Paragraph("2.2 Lexicon", H2)]
story += bl(["<b>VADER</b> (compound, pos, neg): dirancang untuk teks pendek, memanfaatkan kapitalisasi dan tanda baca, sehingga cocok untuk judul berita. Kelemahannya adalah kamus umum, bukan finansial.",
             "<b>Loughran-McDonald</b> (proporsi kata negative, positive, uncertainty, litigious): kamus khusus keuangan yang mengoreksi kata yang bermakna berbeda dalam konteks finansial. Dinormalisasi dengan jumlah token agar tidak bias oleh panjang teks. Kategori <i>uncertainty</i> relevan karena ketidakpastian geopolitik sering menggerakkan mata uang safe-haven."])
story += [Paragraph("2.3 TF-IDF dan Bag-of-Words", H2),
          p("Satu dokumen per hari target dibentuk dari gabungan teks bersih (lowercase, tanpa stopword, lemmatized) semua artikel hari itu, kemudian divektorisasi dengan TF-IDF (unigram + bigram, min_df 5, max_df 0,8, maksimum 5.000 fitur, sublinear tf) atau BoW (hitungan). Karena hanya ada 720 hari train, matriks berdimensi 5.000 dikompresi dengan TruncatedSVD menjadi 50 komponen (fit di train) untuk mengurangi overfitting. TF-IDF meredam kata yang muncul setiap hari; BoW dipakai sebagai pembanding sederhana.")]
story += [Paragraph("2.4 Agregasi harian dan fitur lag", H2),
          p("Model bekerja per hari, sehingga skor artikel diagregasi menjadi mean, standar deviasi, minimum, dan maksimum skor, proporsi artikel negatif/positif, jumlah artikel, serta komposisi delapan kategori kata kunci. Fitur lag-1 dan rata-rata bergulir 3 hari ditambahkan untuk menangkap efek berita yang tertunda. Statistik yang dinormalisasi (mean atau proporsi) dipilih agar hari dengan lonjakan volume (hingga 403 artikel) tidak mendistorsi model.")]

# ---------------- 3
story += [Paragraph("3. Diagram Arsitektur Pipeline", H1)]
img = Image(str(ROOT / "report/pipeline.png")); r = 17*cm / img.imageWidth; img.drawWidth, img.drawHeight = 17*cm, img.imageHeight * r
story += [img, Paragraph("Gambar 1. Arsitektur pipeline: dari data Tugas 1 hingga evaluasi. Kotak merah muda menandai titik pencegahan data leakage.", CAP), Spacer(1, 6)]
story += [tbl([["Modul (src/)", "Fungsi"],
               ["config.py, dataset.py", "Path dan konstanta; penggabungan target, fitur kurs (lag), dan fitur NLP harian"],
               ["split.py", "Definisi target, penautan berita ke hari T, split kronologis"],
               ["preprocessing.py", "Pembersihan teks (versi ringan dan versi bersih), varian title/title+keypoints"],
               ["features.py", "VADER, Loughran-McDonald, agregasi harian + lag, TF-IDF/BoW + SVD (fit di train)"],
               ["baseline.py", "Majority, persistence, ARIMA (order via AIC di train)"],
               ["combined_model.py", "LogReg L2 dan XGBoost; tuning dengan TimeSeriesSplit; protokol val/test"],
               ["evaluate.py, run_experiments.py", "Metrik, uji binomial/bootstrap/McNemar, ablation, penyimpanan hasil"],
               ["export_splits.py", "Ekspor data/train.csv, val.csv, test.csv dan berkas artikel per split"]], [5.2*cm, 11.8*cm])]

# ---------------- 4
story += [Paragraph("4. Hasil Eksperimen Baseline Awal", H1)]
story += [p("Tabel 1 merangkum hasil di data test (241 hari). Eksperimen lengkap (ablation delapan set fitur dengan dua learner) ada di <i>notebook/02_baseline_experiments.ipynb</i> dan <i>results/metrics.csv</i>.")]
order = ["majority", "persistence", "arima"] + [f"{k}:{e}" for e in ["kurs", "kurs+volcat", "kurs+lex_title", "kurs+lex_title_kp", "kurs+tfidf_title", "kurs+tfidf_title_kp", "kurs+bow_title", "kurs+lex+tfidf", "nlp_only(lex_title)"] for k in ("logreg", "xgb")]
rows = [["Model", "DA val", "DA test (CI 95%)", "F1 macro", "MCC", "AUC", "% prediksi 1", "p vs majority"]]
for k in order:
    t, v = test.loc[k], val.loc[k]
    rows.append([k, f(v.DA), f"{f(t.DA)} ({f(t.DA_lo,2)}-{f(t.DA_hi,2)})", f(t.F1_macro), f(t.MCC), f(t.AUC), f"{t.share_pred1*100:.0f}%", f(t.p_vs_majority, 2)])
story += [tbl(rows, [4.6*cm, 1.4*cm, 3.2*cm, 1.5*cm, 1.4*cm, 1.4*cm, 1.7*cm, 1.8*cm]),
          Paragraph("Tabel 1. Hasil evaluasi. Majority baseline test = 0,573. '% prediksi 1' adalah porsi prediksi 'rupiah melemah'; nilai mendekati 100% berarti model hampir selalu menebak kelas mayoritas.", CAP), Spacer(1, 6)]
arima_order = test.loc["arima", "note"]
story += [Paragraph("4.1 Temuan", H2)]
story += bl([f"<b>Tidak ada model yang berbeda signifikan dari majority baseline</b> di data test (p uji binomial &ge; 0,42; CI 95% DA sekitar &plusmn;6 poin). ARIMA ({arima_order}) dan persistence berada di sekitar 0,52-0,53, konsisten dengan autokorelasi return lag-1 yang kecil (0,10).",
             "<b>Fitur NLP tidak meningkatkan model kurs-saja</b>: uji McNemar tidak menunjukkan perbaikan signifikan untuk set fitur mana pun, dan AUC berada di sekitar 0,5. Kombinasi LogReg dengan kurs+lexicon+TF-IDF (81 fitur) justru turun ke DA 0,473 (McNemar p = 0,02 terhadap LogReg kurs-saja), indikasi overfitting pada dimensi tinggi.",
             "<b>DA yang tampak tinggi menyesatkan</b>: LogReg dengan regularisasi kuat memilih C sangat kecil sehingga hampir selalu menebak kelas mayoritas (mis. LogReg kurs+TF-IDF title+keypoints: DA 0,581 tetapi 99% prediksi kelas 1, F1 macro 0,385). Karena itu F1 macro dan MCC wajib dibaca bersama DA.",
             "<b>Analisis eksplorasi</b> (notebook 01): korelasi Spearman sentimen harian dengan return hampir nol; hanya proporsi kata negatif Loughran-McDonald yang nominal signifikan (rho sekitar -0,08, p sekitar 0,01), tidak lolos koreksi Bonferroni dan berlawanan dengan arah yang diharapkan."])
story += [Paragraph("4.2 Interpretasi dan keterbatasan", H2)]
story += bl(["Sinyal harian dari judul/keypoints CNBC lemah dibanding noise kurs harian. Ini hasil awal yang wajar untuk prediksi arah kurs harian, dan hasil negatif ini adalah baseline yang jujur untuk perbandingan pada Tugas 3.",
             "Ukuran sampel kecil (720 hari train, 241 hari test) membatasi kemampuan mendeteksi efek kecil; proporsi kelas juga bergeser antar split (51% di validasi, 57% di test).",
             "Tanggal berita tidak memuat jam terbit sehingga penautan harus konservatif (pergeseran satu hari); jam terbit akan memungkinkan penautan yang lebih presisi.",
             "Dataset hanya berasal dari satu sumber (CNBC, berbahasa Inggris dan berorientasi AS); VADER berbasis kamus umum sehingga kurang peka terhadap istilah finansial.",
             "Banyak konfigurasi diuji pada test yang sama (untuk ablation); p-value tidak dikoreksi untuk perbandingan berganda sehingga temuan bersifat eksploratif."])
story += [Paragraph("4.3 Rencana lanjutan (Tugas 3)", H2),
          p("Arah perbaikan yang diusulkan: agregasi berbasis kejadian/topik (mis. LDA atau NMF) alih-alih rata-rata sentimen, target alternatif (volatilitas atau return besar saja), horizon lebih panjang (arah mingguan), penambahan variabel eksogen (indeks dolar, suku bunga), dan pemilihan fitur yang lebih agresif untuk mengurangi dimensi.")]

doc = SimpleDocTemplate(str(ROOT / "report/Laporan_Tugas2_NLP.pdf"), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="Laporan Tugas 2 NLP - Proposal Pipeline dan Baseline")
def footer(c, d): c.setFont("Helvetica", 8); c.setFillColor(colors.grey); c.drawCentredString(A4[0]/2, 1*cm, f"Halaman {d.page}")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("ok")
