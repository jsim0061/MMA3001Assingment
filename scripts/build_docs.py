"""Generate HTML documentation for reusable modules with pdoc."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,'-m','pdoc',str(root/'src'/'occupancy_prediction'),'-o',str(root/'docs')],check=True)
