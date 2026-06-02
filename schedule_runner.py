import schedule
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_my_zenml_pipeline():
    logging.info("⏰ It's time! Automatically triggering the Customer Churn Pipeline...")
    # The pipeline start command is as if you typed it in the terminal.
    os.system("python -m src.pipeline")


# The pipeline scheduler works every day at 12 midnight (or whatever time it is)
# For a quick test now: Let it ring every minute to see it working on its own!
schedule.every(1).minutes.do(run_my_zenml_pipeline)

logging.info("🚀 Scheduler is active and running in the background. Waiting for the trigger...")

while True:
    schedule.run_pending()
    time.sleep(1)