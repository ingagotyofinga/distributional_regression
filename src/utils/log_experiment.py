from pathlib import Path
import csv

def log_experiment(record, log_file):
    log_file = Path(log_file)
    is_new_file = not log_file.exists()
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=record.keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(record)