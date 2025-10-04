import csv
import os

def save_to_csv(rows, filename="responses.csv"):
    headers = ["Channel", "Reward", "Timestamp"]
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

