from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for script in ["run_financial_distress.py", "run_annual_report_nlp.py"]:
    subprocess.run([sys.executable, str(root / "src" / script)], check=True)

