try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import sys
import runpy

# Get absolute path of the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add Code folder to sys.path
CODE_DIR = os.path.join(ROOT_DIR, "Code")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Important: This allows the app to find 'pipeline', 'src' etc.
os.chdir(ROOT_DIR)

if __name__ == "__main__":
    # Path to the premium dashboard
    DASHBOARD_PATH = os.path.join(CODE_DIR, "app", "premium_dashboard.py")
    
    # Run the dashboard script
    runpy.run_path(DASHBOARD_PATH, run_name="__main__")
