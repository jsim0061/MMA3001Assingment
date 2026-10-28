"""Run the three alternative regression models; original Decision Tree is separate."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['historical_average.py','linear_regression.py','support_vector_regression.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
