"""
FinGuard AI — Application Entry Point Redirect

Executes Home.py to present the 'Home' tab in Streamlit navigation.
"""

import runpy
from pathlib import Path

home_path = Path(__file__).parent / "Home.py"
runpy.run_path(str(home_path), run_name="__main__")
