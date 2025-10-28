# --- Clone dein GitHub-Repository ---
!git clone https://github.com/GassnerChristoph/Code.git
%cd Code

# --- Installiere benötigte Pakete ---
!pip install -q gdown pandas numpy matplotlib

# --- Lade die Daten von öffentlichem Google Drive ---
import gdown

# IDs deiner beiden CSV-Dateien:
id_backscatter = "1XobbLRaWrPsWE1FHDcAFQWGz-CpIreDW"
id_transmission = "1I4-G6H5kvFTSR_BDoRZJxooL8v4CKT43"

gdown.download(id=id_backscatter, output="data_backscatter_realistic_108particles_100ppm.csv", quiet=False)
gdown.download(id=id_transmission, output="data_transmission_108particles_100ppm.csv", quiet=False)

# --- Führe dein Python-Skript aus ---
!python Fig1d.py
