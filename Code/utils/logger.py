import logging
import os
from datetime import datetime
import pytz

# Configure IST timezone
IST = pytz.timezone('Asia/Kolkata')

def ist_time_converter(*args):
    # This is used by logging.Formatter to get the time in IST
    return datetime.now(IST).timetuple()

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate log filename using IST
current_ist_time = datetime.now(IST)
LOG_FILE = os.path.join(LOGS_DIR, f"log_{current_ist_time.strftime('%Y-%m-%d')}.log")

# Setup logging with custom IST converter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Console handler for real-time visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Override the converter for the formatter to use IST
logging.Formatter.converter = ist_time_converter

# Configure root logger to capture all logs
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Set logging level for third-party libraries to capture HTTP requests
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger