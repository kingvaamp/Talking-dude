import webview
import subprocess
import time
import os
import signal
import sys
import multiprocessing

# Configuration
STREAMLIT_FILE = "Talking_Dude.py"
PORT = 8501

def start_streamlit():
    """Lancer le serveur Streamlit en arrière-plan."""
    print(f"🚀 Nettoyage des processus fantômes...")
    os.system("pkill -f 'streamlit run'")
    time.sleep(0.5)

    print(f"🚀 Lancement de Streamlit ({STREAMLIT_FILE})...")
    
    if getattr(sys, 'frozen', False):
        # We are running as a PyInstaller bundle
        return subprocess.Popen([sys.executable, "run_streamlit"])
    else:
        # We are running from normal python
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
            proc.terminate()
            sys.exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if len(sys.argv) > 1 and sys.argv[1] == "run_streamlit":
        # Internal Streamlit launcher for PyInstaller
        import streamlit.web.cli as stcli
        
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
            
        script_path = os.path.join(base_path, "Talking_Dude.py")
        sys.argv = ["streamlit", "run", script_path, 
                    "--server.port", str(PORT), 
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"]
        sys.exit(stcli.main())
    else:
        main()
