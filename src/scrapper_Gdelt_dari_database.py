import re
import requests
import os

# 1. Konfigurasi Rentang Tanggal (Format: YYYYMMDDHHMMSS)
START_DATE = "20210901000000"  # 1 September 2021 jam 00:00:00
END_DATE   = "20260901235959"  # 1 September 2026 jam 23:59:59

# Pilih jenis file yang ingin diunduh: 'export', 'mentions', atau 'gkg'
# Set None jika ingin mengambil semua tipe
TARGET_TYPE = "mentions" 

MASTER_LIST_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
SAVE_DIR = r"C:\Raditya\Raditya Stuff\NLP\Tugas 1\geopolitic-sentiment-analysis\data\raw\Gdelt_mentions"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Mendownload dan membaca masterfilelist.txt...")

# 2. Ambil masterfilelist
response = requests.get(MASTER_LIST_URL, stream=True)

target_urls = []

# Pattern regex untuk mengekstrak timestamp dari URL GDELT
# Contoh URL: http://data.gdeltproject.org/gdeltv2/20200901000000.export.csv.zip
pattern = re.compile(r'/(\d{14})\.([a-z]+)\.csv\.zip')

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        parts = line_str.strip().split()
        
        if len(parts) >= 3:
            file_url = parts[2]
            match = pattern.search(file_url)
            
            if match:
                timestamp = match.group(1) # YYYYMMDDHHMMSS
                file_type = match.group(2) # export, mentions, atau gkg
                
                # Filter berdasarkan Tanggal
                if START_DATE <= timestamp <= END_DATE:
                    # Filter berdasarkan Tipe File (jika ditentukan)
                    if TARGET_TYPE is None or file_type == TARGET_TYPE:
                        target_urls.append(file_url)

print(f"Ditemukan {len(target_urls)} file yang sesuai kriteria.")

# 3. Contoh memproses / mendownload file
# Catatan: Karena jumlahnya sangat banyak, ganti logika ini dengan download parallel/multithreading
for idx, url in enumerate(target_urls[:10]): # Contoh download 10 file pertama dulu
    filename = url.split('/')[-1]
    filepath = os.path.join(SAVE_DIR, filename)
    
    print(f"[{idx+1}/{len(target_urls)}] Mendownload {filename}...")
    file_resp = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(file_resp.content)

print("Selesai!")