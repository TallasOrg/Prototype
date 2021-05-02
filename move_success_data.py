import os
import csv
from loguru import logger

def import_data():
    guide_strands = []
    with open('data/HueskenRNA_retry.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            guide_strands.append(row)

    with open('success_data.txt', 'r') as f:
        success_data_string = f.read()
    success_data = [line for line in success_data_string.split('\n') if line]

    with open('missing_data.txt', 'r') as f:
        missing_data_string = f.read()
    missing_data = [line for line in missing_data_string.split('\n') if line]

    with open('missing_rid.txt', 'r') as f:
        missing_rid_string = f.read()
    missing_rid = [line for line in missing_rid_string.split('\n') if line]

    return guide_strands, success_data, missing_data, missing_rid

def write_huesken_data(filename, data):
    headers = ['guide_strand', 'norm_inhibitory_activity', 'free_energy']
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    huseken_original_data, success_strands, missing_data, missing_rid_strands = import_data()
    huseken_original_strands = [row[0] for row in huseken_original_data]
    missing_data_strands = [row.split(',')[0] for row in missing_data]
    huesken_success = []
    huesken_retry = []
    for i, strand in enumerate(huseken_original_strands):
        if strand in success_strands:
            huesken_success.append(huseken_original_data[i])
        else:
            huesken_retry.append(huseken_original_data[i])
    write_huesken_data('data/HueskenRNA_success2.csv', huesken_success)
    write_huesken_data('data/HueskenRNA_retry2.csv', huesken_retry)