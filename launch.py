import webview
import subprocess
import time
import os
import signal
import sys

# Configuration
STREAMLIT_FILE = "Talking_Dude.py"
PORT = 8501

def start_streamlit():
    """Lancer le serveur Streamlit en arrière-plan."""
    print(f"🚀 Nettoyage des processus fantômes...")
    os.system("pkill -f 'streamlit run'")
    time.sleep(0.5)

    print(f"🚀 Lancement de Streamlit ({STREAMLIT_FILE})...")
    # Utiliser sys.executable pour plus de robustesse
    python_exe = sys.executable
    return subprocess.Popen([
        python_exe, "-m", "streamlit", "run", STREAMLIT_FILE,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.runOnSave", "true",
        "--browser.gatherUsageStats", "false"
    ])

def main():
    proc = None
    try:
        # 1. Lancer Streamlit
        proc = start_streamlit()
        
        # 2. Attendre que le serveur soit prêt (on pourrait faire un check HTTP mais un sleep suffit souvent)
        print("⏳ Attente du serveur (5s)...")
        time.sleep(5)
        
        # 3. Créer la fenêtre WebView
        print("🌐 Ouverture de la fenêtre native...")
        window = webview.create_window(
            "🎙️ Talking Dude — Live Interpreter",
            f"http://localhost:{PORT}",
            width=1280,
            height=850,
            resizable=True,
            confirm_close=True
        )
        
        # 4. Lancer la boucle WebView
        webview.start()
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        # 5. Nettoyage à la fermeture
        if proc:
            print("🛑 Fermeture du serveur Streamlit...")
            os.kill(proc.pid, signal.SIGTERM)
            sys.exit(0)

if __name__ == "__main__":
    main()
