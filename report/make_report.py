"""Membangun laporan PDF Tugas 2 dari results/metrics.csv dan report/pipeline.png."""
from math import sqrt
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

# Isi sebelum membuat laporan
GROUP = "[isi nama kelompok]"
MEMBERS = "[isi nama dan NIM anggota]"
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


def pct(x): return f"{x*100:.1f}".replace(".", ",") + "%"

story = []
story += [Paragraph("Proposal Pipeline &amp; Eksperimen Baseline", ParagraphStyle("t", parent=H1, fontSize=19, spaceAfter=2)),
          Paragraph("Prediksi Arah Nilai Tukar USD/IDR dari Berita Geopolitik CNBC dengan NLP Klasik", ParagraphStyle("st", parent=B, fontSize=11.5, leading=15, alignment=0)),
          Paragraph(f"Proyek NLP - Tugas 2 &nbsp;|&nbsp; Kelompok: {GROUP} &nbsp;|&nbsp; Anggota: {MEMBERS}", CAP),
          Spacer(1, 6)]

# ---------------- 1
story += [Paragraph("1. Formulasi Tugas", H1)]
story += [p("Hipotesis proyek adalah bahwa berita geopolitik global membantu memprediksi pergerakan kurs USD. Pada Tugas 2 hipotesis ini dioperasionalkan menjadi tugas klasifikasi biner harian dan diuji dengan pipeline NLP klasik (lexicon dan vektorisasi TF-IDF) tanpa model pre-trained atau transformer.")]
story += [Paragraph("1.1 Data", H2),
          p("Data berasal dari Tugas 1: 41.406 artikel CNBC (1 Sep 2021 - 1 Sep 2026) yang sudah difilter ke topik geopolitik dan makro (delapan kategori kata kunci). Kurs yang dipakai adalah JISDOR Bank Indonesia harian. Setelah penautan tanggal diperoleh 1.201 hari kerja target, dengan rata-rata 34 artikel per hari.")]
story += [Paragraph("1.2 Definisi target", H2),
          p("Unit prediksi adalah hari kerja <b>T</b>. Target <b>y<sub>T</sub> = 1</b> jika kurs<sub>T</sub> &gt; kurs<sub>T-1</sub> (rupiah melemah terhadap USD), dan <b>y<sub>T</sub> = 0</b> jika sebaliknya. Terdapat 13 hari dengan kurs tidak berubah (1,1%) yang digabungkan ke kelas 0 agar deret hari tetap utuh dan fitur lag tidak terputus. Proporsi kelas 1 adalah 55,4% untuk seluruh data (train 56,1%, validasi 51,2%, test 57,3%).")]
story += [p("Klasifikasi biner dipilih dibanding regresi level kurs karena (i) kurs bersifat non-stasioner (14.284 menjadi 17.727 IDR/USD) sehingga level mudah diprediksi lewat tren tanpa ada hubungan dengan berita, (ii) hipotesis proyek berbicara tentang arah pergerakan, dan (iii) metrik arah mudah ditafsirkan dan dibandingkan dengan baseline.")]
story += [Paragraph("1.3 Penautan berita ke target dan pencegahan data leakage", H2),
          p("Data tanggal CNBC hanya berisi tanggal (tanpa jam) dan mengikuti waktu AS, sehingga berita bertanggal d baru tercermin di pasar Asia pada hari kerja berikutnya. Aturan yang dipakai: <b>berita bertanggal d dipasangkan ke T, yaitu hari kerja pertama yang benar-benar setelah d</b>. Akibatnya berita Jumat, Sabtu, dan Minggu semuanya dipakai untuk memprediksi Senin, dan berita pada hari libur nasional dipakai untuk hari kerja berikutnya. Label kurs hari yang sama dengan berita (seperti kolom <i>kurs_direction</i> pada keluaran Tugas 1) tidak dipakai karena berita dan kurs pada tanggal yang sama dapat saling tumpang tindih waktu.")]
story += bl(["<b>Fitur kurs</b> untuk hari T hanya memakai data sampai T-1 (semua lag dibuat dengan shift).",
             "<b>Vocabulary, IDF, dan SVD</b> dari TF-IDF di-fit hanya pada hari train, lalu diterapkan ke validasi dan test.",
             "<b>Split kronologis tanpa shuffle</b>; hyperparameter dipilih dengan data validasi; data test dipakai satu kali."])
story += [Paragraph("1.4 Pembagian data", H2)]
story += [tbl([["Split", "Periode target", "Jumlah hari", "P(y=1)"],
               ["Train (60%)", "2 Sep 2021 - 22 Agu 2024", "720", "0,561"],
               ["Validasi (20%)", "23 Agu 2024 - 28 Agu 2025", "240", "0,512"],
               ["Test (20%)", "29 Agu 2025 - 1 Sep 2026", "241", "0,573"]], [3.6*cm, 6*cm, 3*cm, 2.6*cm]), Spacer(1, 4),
          p("Semua model dilatih pada data train. Data validasi dipakai untuk memilih hyperparameter dan data test dipakai satu kali untuk skor akhir. Berkas hasil pembagian ada di <i>data/train.csv, val.csv, test.csv</i> (matriks fitur harian) dan <i>data/splits/*_articles.csv</i> (level artikel).")]
story += [Paragraph("1.5 Metrik evaluasi", H2)]
story += bl(["<b>Directional Accuracy (DA)</b>: proporsi arah yang benar; dibandingkan dengan <b>majority baseline</b> (kelas mayoritas train) dan <b>persistence</b> (arah hari T sama dengan T-1).",
             "<b>F1 macro</b>: rata-rata F1 kelas 0 dan kelas 1. Model yang hanya menebak kelas mayoritas bisa punya DA lumayan tinggi tetapi F1 macro rendah, sehingga kedua metrik dibaca bersama. Kami juga melaporkan porsi hari yang diprediksi kelas 1 untuk melihat model yang hampir selalu menebak satu kelas.",
             "Dengan 241 hari test, satu standar error DA sekitar 3 poin, sehingga selisih di bawah sekitar 6 poin tidak dapat dibedakan dari kebetulan."])

# ---------------- 2
story += [Paragraph("2. Alasan Pemilihan Pendekatan Ekstraksi Fitur NLP", H1)]
story += [p("Ketentuan tugas melarang embedding pre-trained dan model berbasis transformer, sehingga seluruh fitur teks dibangun dari lexicon dan vektorisasi klasik. Dua keluarga fitur dipilih karena saling melengkapi: lexicon memberi sinyal <i>sentimen</i> berdimensi rendah, sedangkan TF-IDF menangkap <i>topik dan kosakata</i> tanpa asumsi kamus.")]
story += [Paragraph("2.1 Unit teks: judul", H2),
          p("Judul tersedia untuk 100% artikel, singkat, dan ditulis editor untuk merangkum kejadian, sehingga dipakai sebagai teks utama. Keypoints (ringkasan editor CNBC) memberi konteks lebih kaya tetapi hanya ada pada 51,6% artikel dan cakupannya berubah antar tahun (sekitar 46% sampai 62%), yang dapat menimbulkan pergeseran distribusi fitur. Karena itu keypoints tidak dipakai untuk model; pada notebook EDA sinyal sentimen judul dan judul+keypoints dibandingkan secara deskriptif. Isi artikel penuh tidak tersedia dalam dataset dan biaya scraping ulang tinggi, dan ekstraksi entitas tidak dilakukan agar pipeline tetap sederhana.")]
story += [Paragraph("2.2 Lexicon", H2)]
story += bl(["<b>VADER</b> (compound, pos, neg): dirancang untuk teks pendek, memanfaatkan kapitalisasi dan tanda baca, sehingga cocok untuk judul berita. Kelemahannya adalah kamus umum, bukan finansial.",
             "<b>Loughran-McDonald</b> (proporsi kata negative, positive, uncertainty, litigious): kamus khusus keuangan yang mengoreksi kata yang bermakna berbeda dalam konteks finansial. Dinormalisasi dengan jumlah token agar tidak bias oleh panjang teks. Kategori <i>uncertainty</i> relevan karena ketidakpastian geopolitik sering menggerakkan mata uang safe-haven."])
story += [Paragraph("2.3 TF-IDF", H2),
          p("Satu dokumen per hari target dibentuk dari gabungan teks bersih (lowercase, tanpa stopword, lemmatized) semua artikel hari itu, kemudian divektorisasi dengan TF-IDF (unigram + bigram, min_df 5, max_df 0,8, maksimum 5.000 fitur, sublinear tf). TF-IDF dipilih dibanding Bag-of-Words biasa karena meredam kata yang muncul hampir setiap hari. Karena hanya ada 720 hari train, matriks berdimensi 5.000 dikompresi dengan TruncatedSVD menjadi 50 komponen (fit di train) untuk mengurangi overfitting.")]
story += [Paragraph("2.4 Agregasi harian", H2),
          p("Model bekerja per hari, sehingga skor lexicon tiap artikel diagregasi menjadi rata-rata skor VADER compound, proporsi artikel negatif dan positif, serta rata-rata proporsi kata Loughran-McDonald. Rata-rata dan proporsi dipilih agar hari dengan lonjakan jumlah artikel (hingga 403 artikel) tidak mendistorsi model.")]
story += [Paragraph("2.5 Pemilihan model", H2),
          p("Baseline time-series terdiri dari <b>naive</b> (majority dan persistence), <b>ARIMA</b>(1,0,1) pada return kurs, dan <b>XGBoost</b> dengan fitur kurs saja. Model gabungan memakai <b>XGBoost yang sama</b> dengan fitur kurs ditambah fitur NLP, sehingga satu-satunya perbedaan antara baseline XGBoost dan model gabungan adalah fitur NLP. XGBoost berbasis pohon dapat menangkap hubungan non-linear, tidak memerlukan standardisasi fitur, dan dibuat dangkal serta teregularisasi karena datanya hanya ratusan hari. Hyperparameter (kedalaman pohon dan jumlah pohon) dipilih dari grid kecil dengan akurasi arah di data validasi.")]

# ---------------- 3
story += [Paragraph("3. Diagram Arsitektur Pipeline", H1)]
img = Image(str(ROOT / "report/pipeline.png")); r = 17*cm / img.imageWidth; img.drawWidth, img.drawHeight = 17*cm, img.imageHeight * r
story += [img, Paragraph("Gambar 1. Arsitektur pipeline: dari data Tugas 1 hingga evaluasi. Kotak merah muda menandai titik pencegahan data leakage.", CAP), Spacer(1, 6)]
story += [tbl([["Modul (src/)", "Fungsi"],
               ["config.py, dataset.py", "Path dan konstanta; penggabungan target, fitur kurs (lag), dan fitur NLP harian"],
               ["split.py", "Definisi target, penautan berita ke hari T, split kronologis"],
               ["preprocessing.py", "Pembersihan teks (versi ringan untuk lexicon, versi bersih untuk TF-IDF)"],
               ["features.py", "VADER, Loughran-McDonald, agregasi harian, TF-IDF + SVD (fit di train)"],
               ["baseline.py", "Majority, persistence, ARIMA"],
               ["model.py", "XGBoost (fitur kurs saja atau kurs + NLP); pemilihan hyperparameter dengan data validasi"],
               ["evaluate.py, run_experiments.py", "Directional Accuracy, F1 macro, confusion matrix; menjalankan semua eksperimen"],
               ["export_splits.py", "Ekspor data/train.csv, val.csv, test.csv dan berkas artikel per split"]], [5.2*cm, 11.8*cm])]

# ---------------- 4
story += [Paragraph("4. Hasil Eksperimen Baseline Awal", H1)]
n_test = int(test.loc["majority", "n"])
maj = test.loc["majority", "DA"]
se = sqrt(maj * (1 - maj) / n_test)
story += [p(f"Tabel 1 merangkum hasil eksperimen. Skor akhir dihitung pada data test ({n_test} hari) yang hanya dipakai satu kali. Hasil lengkap ada di <i>notebook/02_baseline_experiments.ipynb</i> dan <i>results/metrics.csv</i>.")]
labels = {"majority": "Majority (naive)", "persistence": "Persistence (naive)", "arima": "ARIMA(1,0,1)",
          "xgb:kurs": "XGBoost: kurs saja (baseline)", "xgb:kurs+lexicon": "XGBoost: kurs + lexicon (gabungan)",
          "xgb:kurs+tfidf": "XGBoost: kurs + TF-IDF (gabungan)", "xgb:kurs+lexicon+tfidf": "XGBoost: kurs + lexicon + TF-IDF (gabungan)"}
order = list(labels)
rows = [["Model", "DA val", "DA test", "F1 macro test", "% prediksi kelas 1 (test)"]]
for k in order:
    rows.append([labels[k], f(val.loc[k, "DA"]), f(test.loc[k, "DA"]), f(test.loc[k, "F1_macro"]), f"{test.loc[k, 'share_pred1']*100:.0f}%"])
story += [tbl(rows, [6.6*cm, 2*cm, 2*cm, 2.8*cm, 3.6*cm]),
          Paragraph(f"Tabel 1. Hasil evaluasi. DA = Directional Accuracy. Majority baseline test = {f(maj)}. '% prediksi kelas 1' adalah porsi prediksi 'rupiah melemah'; nilai mendekati 100% berarti model hampir selalu menebak kelas mayoritas.", CAP), Spacer(1, 6)]

# temuan otomatis dari angka di metrics.csv
gab = [k for k in order if k.startswith("xgb:kurs+")]
xkurs = test.loc["xgb:kurs", "DA"]
ref = max(maj, xkurs)
finds = [f"<b>Baseline</b>: majority {f(maj)}, persistence {f(test.loc['persistence','DA'])}, ARIMA {f(test.loc['arima','DA'])}, dan XGBoost kurs-saja {f(xkurs)} (DA test)."]
gtxt = "; ".join(f"{labels[k].split(': ')[1].split(' (')[0]} {f(test.loc[k,'DA'])}" for k in gab)
finds.append(f"<b>Model gabungan</b> (DA test): {gtxt}.")
best = max(gab, key=lambda k: test.loc[k, "DA"])
gap = test.loc[best, "DA"] - ref
if gap > 2 * se:
    finds.append(f"Model gabungan terbaik ({labels[best].split(': ')[1].split(' (')[0]}) unggul {gap*100:.1f} poin dari baseline terkuat, melebihi sekitar dua standar error ({2*se*100:.1f} poin), tetapi hasil dari satu split test perlu ditafsirkan hati-hati.")
else:
    finds.append(f"Tidak ada model gabungan yang mengungguli baseline terkuat (majority atau XGBoost kurs-saja) dengan selisih lebih dari sekitar dua standar error ({2*se*100:.1f} poin), sehingga belum ada bukti yang jelas bahwa fitur NLP membantu memprediksi arah kurs.")
degen = [k for k in order if test.loc[k, "share_pred1"] >= 0.9 or test.loc[k, "share_pred1"] <= 0.1]
if degen:
    finds.append("Model yang hampir selalu menebak satu kelas (" + ", ".join(labels[k] for k in degen) + ") memiliki DA yang tidak mencerminkan kemampuan prediksi; F1 macro dan porsi prediksi kelas 1 perlu dibaca bersama DA.")
story += [Paragraph("4.1 Temuan", H2)] + bl(finds)
story += [Paragraph("4.2 Interpretasi dan keterbatasan", H2)]
story += bl(["Ukuran sampel kecil (720 hari train, 241 hari test) membatasi kemampuan mendeteksi efek kecil; proporsi kelas juga bergeser antar split (51% di validasi, 57% di test).",
             "Tanggal berita tidak memuat jam terbit sehingga penautan harus konservatif (pergeseran satu hari); jam terbit akan memungkinkan penautan yang lebih presisi.",
             "Dataset hanya berasal dari satu sumber (CNBC, berbahasa Inggris dan berorientasi AS); VADER berbasis kamus umum sehingga kurang peka terhadap istilah finansial.",
             "Hasil ini adalah baseline awal. Prediksi arah kurs harian sulit, sehingga hasil yang mendekati majority baseline wajar dan menjadi pembanding untuk Tugas 3."])
story += [Paragraph("4.3 Rencana lanjutan (Tugas 3)", H2),
          p("Arah perbaikan yang dapat dicoba: agregasi berbasis topik/kejadian (mis. LDA atau NMF) alih-alih rata-rata sentimen, horizon prediksi yang lebih panjang (arah mingguan), dan penambahan variabel eksogen (indeks dolar, suku bunga).")]

doc = SimpleDocTemplate(str(ROOT / "report/Laporan_Tugas2_NLP.pdf"), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="Laporan Tugas 2 NLP - Proposal Pipeline dan Baseline")
def footer(c, d): c.setFont("Helvetica", 8); c.setFillColor(colors.grey); c.drawCentredString(A4[0]/2, 1*cm, f"Halaman {d.page}")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("ok")
