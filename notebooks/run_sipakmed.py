
import subprocess
import sys

print("Lancement du dashboard SIPAKMED...")
print("Utilisation de VOS donnees reelles")
print("Ouvrez: http://localhost:8501")

try:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_sipakmed.py"])
except KeyboardInterrupt:
    print("Dashboard ferme")
except Exception as e:
    print(f"Erreur: {e}")
    print("Essayez:")
    print("  streamlit run streamlit_sipakmed.py")
