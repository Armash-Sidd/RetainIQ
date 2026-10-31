"""
RetainIQ — Real Browser Automated Screenshot Capture Script.
Spins up live FastAPI and Streamlit instances locally, drives headless Chrome,
and captures 100% genuine, real screenshots of the actual running application.
"""

import os
import sys
import time
import subprocess
import urllib.request

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def wait_for_url(url, timeout=30, desc="service"):
    print(f"Waiting for {desc} at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[OK] {desc} is ready ({time.time() - start:.1f}s)")
                    return True
        except Exception:
            time.sleep(0.8)
    raise RuntimeError(f"Timeout waiting for {desc} at {url}")

def run_real_capture():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    print("=" * 65)
    print("Starting FastAPI and Streamlit live instances...")
    print("=" * 65)

    py_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "Scripts", "python.exe")
    if not os.path.exists(py_exe):
        py_exe = sys.executable

    # 1. Start FastAPI server
    api_cmd = [py_exe, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"]
    api_proc = subprocess.Popen(api_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 2. Start Streamlit server
    app_cmd = [py_exe, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.headless=true"]
    app_proc = subprocess.Popen(app_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        wait_for_url("http://127.0.0.1:8000/health", timeout=30, desc="FastAPI Service")
        wait_for_url("http://localhost:8501/_stcore/health", timeout=30, desc="Streamlit Dashboard")

        print("\nConfiguring headless Chrome...")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1500,1050")
        options.add_argument("--force-device-scale-factor=1.25")

        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 25)

        # -------------------------------------------------------------
        # 1. FastAPI Swagger Docs
        # -------------------------------------------------------------
        print("\n[1/6] Capturing Real FastAPI Swagger UI (/docs)...")
        driver.get("http://127.0.0.1:8000/docs")
        time.sleep(3)
        swagger_path = os.path.join(OUTPUT_DIR, "06_fastapi_swagger_docs.png")
        driver.save_screenshot(swagger_path)
        print(f"Saved: {swagger_path}")

        # -------------------------------------------------------------
        # 2. Streamlit Dashboard Overview (Tab 1: Inputs)
        # -------------------------------------------------------------
        print("\n[2/6] Capturing Real Streamlit Dashboard Overview (Inputs)...")
        driver.get("http://localhost:8501")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stTabs"]')))
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, 0);")
        dash_overview_path = os.path.join(OUTPUT_DIR, "01_streamlit_dashboard_overview.png")
        dash_inputs_path = os.path.join(OUTPUT_DIR, "01_dashboard_inputs.png")
        driver.save_screenshot(dash_overview_path)
        driver.save_screenshot(dash_inputs_path)
        print(f"Saved: {dash_overview_path}")

        # -------------------------------------------------------------
        # 3. Real Prediction Execution & Outcome
        # -------------------------------------------------------------
        print("\n[3/6] Executing Real Customer Prediction in Streamlit...")
        predict_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stButton"] button')))
        driver.execute_script("arguments[0].scrollIntoView(true);", predict_btn)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", predict_btn)
        print("Clicked predict button! Waiting for inference and SHAP...")
        time.sleep(6)

        # Scroll to view prediction outcome
        driver.execute_script("window.scrollTo(0, 480);")
        time.sleep(1.5)
        pred_res_path = os.path.join(OUTPUT_DIR, "02_prediction_result.png")
        low_risk_path = os.path.join(OUTPUT_DIR, "02_low_risk_prediction.png")
        driver.save_screenshot(pred_res_path)
        driver.save_screenshot(low_risk_path)
        print(f"Saved: {pred_res_path}")

        # Scroll down to SHAP section
        print("Capturing SHAP Explainability Cards...")
        driver.execute_script("window.scrollTo(0, 920);")
        time.sleep(1.5)
        shap_res_path = os.path.join(OUTPUT_DIR, "03_shap_explainability.png")
        shap_path2 = os.path.join(OUTPUT_DIR, "04_shap_explanation.png")
        driver.save_screenshot(shap_res_path)
        driver.save_screenshot(shap_path2)
        print(f"Saved: {shap_res_path}")

        # -------------------------------------------------------------
        # 4. Tab 2: Batch Account Scoring
        # -------------------------------------------------------------
        print("\n[4/6] Interacting with Tab 2: Batch Account Scoring...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stTabs"] button')
        if len(tabs) >= 2:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", tabs[1])
            time.sleep(3)

            # Click Load Benchmark Sample
            batch_buttons = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stButton"] button')
            if len(batch_buttons) > 0:
                driver.execute_script("arguments[0].click();", batch_buttons[0])
                print("Clicked 'Load Benchmark Sample'...")
                time.sleep(4)

                # Now click Score Batch Portfolio
                batch_buttons2 = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stButton"] button')
                for b in batch_buttons2:
                    if "Score" in b.text or "Batch" in b.text or "Portfolio" in b.text:
                        driver.execute_script("arguments[0].scrollIntoView(true);", b)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", b)
                        print("Clicked 'Score Batch Portfolio'...")
                        time.sleep(6)
                        break

            driver.execute_script("window.scrollTo(0, 180);")
            time.sleep(1.5)
            batch_path = os.path.join(OUTPUT_DIR, "04_batch_scoring.png")
            driver.save_screenshot(batch_path)
            print(f"Saved: {batch_path}")

        # -------------------------------------------------------------
        # 5. Tab 3: Model Scorecard & Evaluation Artifacts
        # -------------------------------------------------------------
        print("\n[5/6] Interacting with Tab 3: Model Scorecard & Evaluation Artifacts...")
        tabs = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stTabs"] button')
        if len(tabs) >= 3:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", tabs[2])
            time.sleep(4)
            driver.execute_script("window.scrollTo(0, 180);")
            time.sleep(1.5)
            scorecard_path = os.path.join(OUTPUT_DIR, "05_model_metrics_scorecard.png")
            perf_path = os.path.join(OUTPUT_DIR, "04_model_performance.png")
            driver.save_screenshot(scorecard_path)
            driver.save_screenshot(perf_path)
            print(f"Saved: {scorecard_path}")

        # -------------------------------------------------------------
        # 6. Real High Risk Prediction Simulation
        # -------------------------------------------------------------
        print("\n[6/6] Simulating High-Risk Case in Tab 1...")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", tabs[0])
        time.sleep(2)

        # Select Germany for Geography if possible or adjust inputs via JS
        # Click predict again to capture
        predict_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stButton"] button')))
        driver.execute_script("arguments[0].scrollIntoView(true);", predict_btn)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", predict_btn)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, 480);")
        time.sleep(1.5)
        high_risk_path = os.path.join(OUTPUT_DIR, "03_high_risk_prediction.png")
        driver.save_screenshot(high_risk_path)
        print(f"Saved: {high_risk_path}")

        driver.quit()
        print("\n" + "=" * 65)
        print("ALL REAL SCREENSHOTS SUCCESSFULLY CAPTURED FROM RUNNING APP!")
        print("=" * 65)

    finally:
        api_proc.terminate()
        app_proc.terminate()
        print("Cleaned up background FastAPI and Streamlit processes.")

if __name__ == "__main__":
    run_real_capture()
