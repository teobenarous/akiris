import argparse
import logging
from pathlib import Path

from app.main import process_message
from app.pager import PagerService
from app.predictor import AKIPredictor
from app.state import PatientStore

logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s", "lvl": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger("RecoveryTool")


def read_text_dump(file_path: Path):
    """Parses a plain text HL7 dump into strings for process_message."""
    with open(file_path, "r", encoding="utf-8") as f:
        current_message = []
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip file/batch headers and footers
            if line.startswith(("FSH", "BSH", "FTS", "BTS")):
                continue

            # MSH indicates the start of a new message
            if line.startswith("MSH"):
                if current_message:
                    # Yield as a standard string, NOT bytes
                    yield "\r".join(current_message)
                current_message = [line]
            else:
                if current_message:
                    current_message.append(line)

        # Yield the final message in the file
        if current_message:
            yield "\r".join(current_message)


def run_recovery(file_path: Path):
    logger.info(f"Starting recovery using file: {file_path}")

    # (1) Initialize core components
    store = PatientStore()
    store.hydrate()
    predictor = AKIPredictor()
    pager = PagerService()

    # (2) Extract and process
    messages_processed = 0
    try:
        for raw_msg in read_text_dump(file_path):
            process_message(raw_msg, store, predictor, pager)
            messages_processed += 1
    finally:
        logger.info(f"Successfully processed {messages_processed} missed messages.")

    # (3) Graceful Shutdown & Save
    logger.info("Shutting down pager pool...")
    pager.shutdown()  # Wait for backdated HTTP requests to finish
    logger.info("Saving updated state...")
    store.save()  # Atomically save the new state
    logger.info("Recovery complete.")


def main():
    parser = argparse.ArgumentParser(description="AKI HL7 Data Recovery Tool")
    parser.add_argument(
        "--file", type=Path, required=True, help="Path to text dump file."
    )
    args = parser.parse_args()

    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return

    run_recovery(args.file)


if __name__ == "__main__":
    main()
