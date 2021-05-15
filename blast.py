import os
import csv
import time
import argparse
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from loguru import logger

def log_current_results(guide_strands, success_data, missing_data, missing_rid):
    total_run = len(success_data) + len(missing_data) + len(missing_rid)
    logger.info(f'Processed {total_run} of {len(guide_strands)}')
    logger.info(f'Success: {len(success_data)} of {total_run}')
    logger.warning(f'Missing Data: {len(missing_data)} of {total_run}')
    logger.error(f'No RID: {len(missing_rid)} of {total_run}')

def import_data():
    guide_strands = []
    with open('data/HueskenRNA_retry.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            guide_strands.append(row[0])

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

def get_files_metadata(guide_strand, rid):
    return [
        {
            'filename': f'{guide_strand}.txt',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=Text&FORMAT_OBJECT=Alignment&DESCRIPTIONS=100&ALIGNMENTS=100&CMD=Get&DOWNLOAD_TEMPL=Results_All&ADV_VIEW=on'
        },
        {
            'filename': f'{guide_strand}.xml',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=XML&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}.asn',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=ASN.1&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_json_seq_align.json',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=JSONSA&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_hit_table.txt',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=Text&FORMAT_OBJECT=Alignment&DESCRIPTIONS=100&ALIGNMENT_VIEW=Tabular&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_hit_table.csv',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=CSV&FORMAT_OBJECT=Alignment&DESCRIPTIONS=100&ALIGNMENT_VIEW=Tabular&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_multiple.xml',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RID={rid}&FORMAT_TYPE=XML2&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_single.xml',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=XML2_S&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_multiple.json',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RID={rid}&FORMAT_TYPE=JSON2&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}_single.json',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=JSON2_S&FORMAT_OBJECT=Alignment&CMD=Get'
        },
        {
            'filename': f'{guide_strand}.sam',
            'url': f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?RESULTS_FILE=on&RID={rid}&FORMAT_TYPE=SAM_SQ&FORMAT_OBJECT=Alignment&CMD=Get'
        }
    ]


def blast(starting_strand, iterations=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    guide_strands, success_data, missing_data, missing_rid = import_data()

    starting_index = guide_strands.index(starting_strand)
    ending_index = iterations + starting_index if iterations else len(guide_strands)

    for i in range(starting_index, ending_index):
        guide_strand = guide_strands[i]
        logger.debug(f'Processing strand: {guide_strand}')

        resp = requests.get(f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=off&match_scores=1,-3&expect=1000&word_size=7&gapcosts=5&DATABASE=nt&FORMAT_TYPE=JSON2&QUERY={guide_strand}')
        soup = BeautifulSoup(resp.text, 'html.parser')
        if not soup:
            missing_rid.append(guide_strand)
            with open('missing_rid.txt', 'w') as f:
                missing_string = '\n'.join(missing_rid)
                f.write(f'{missing_string}')
            log_current_results(guide_strands, success_data, missing_data, missing_rid)
            continue
        rid = soup.find(attrs={"name": "RID"}).get('value')
        if not rid:
            missing_rid.append(guide_strand)
            with open('missing_rid.txt', 'w') as f:
                missing_string = '\n'.join(missing_rid)
                f.write(f'{missing_string}')
            log_current_results(guide_strands, success_data, missing_data, missing_rid)
        else:
            save_dir = dir_path + f'/data/blast/{guide_strand}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            data_files = get_files_metadata(guide_strand, rid)
            time.sleep(60)
            for data_file in data_files:
                r = requests.get(data_file['url'])
                with open(f'{save_dir}/{data_file["filename"]}', 'wb') as f:
                    f.write(r.content)
            
            with open(f'{save_dir}/{data_files[0]["filename"]}', 'r') as f:
                file_string = f.read()
            
            if '!DOCTYPE' in file_string:
                missing_data.append(f'{guide_strand}, {rid}')
                with open('missing_data.txt', 'w') as f:
                    missing_string = '\n'.join(missing_data)
                    f.write(f'{missing_string}')
                log_current_results(guide_strands, success_data, missing_data, missing_rid)
            else:
                success_data.append(guide_strand)
                with open('success_data.txt', 'w') as f:
                    success_string = '\n'.join(success_data)
                    f.write(f'{success_string}')
                log_current_results(guide_strands, success_data, missing_data, missing_rid)

def retry():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    guide_strands, success_data, missing_data, missing_rid = import_data()
    new_missing_data = missing_data
    logger.debug('Before retry results')
    log_current_results(guide_strands, success_data, missing_data, missing_rid)
    for line in missing_data:
        guide_strand = line.split(',')[0].strip()
        rid = line.split(',')[1].strip()
        save_dir = dir_path + f'/data/blast/{guide_strand}'
        data_files = get_files_metadata(guide_strand, rid)
        logger.info(f'Downloading files for guide_strand: {guide_strand}, rid: {rid}')
        for data_file in data_files:
            r = requests.get(data_file['url'])
            with open(f'{save_dir}/{data_file["filename"]}', 'wb') as f:
                f.write(r.content)
        
        with open(f'{save_dir}/{data_files[0]["filename"]}', 'r') as f:
            file_string = f.read()
        
        if '!DOCTYPE' in file_string:
            continue
        else:
            success_data.append(guide_strand)
            new_missing_data.remove(line)
        time.sleep(10)
    

    with open('missing_data.txt', 'w') as f:
        missing_string = '\n'.join(new_missing_data)
        f.write(f'{missing_string}')
    
    with open('success_data.txt', 'w') as f:
        success_string = '\n'.join(success_data)
        f.write(f'{success_string}')
    
    logger.debug('After retry results')
    log_current_results(guide_strands, success_data, new_missing_data, missing_rid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to run blast queries against https://blast.ncbi.nlm.nih.gov API')
    parser.add_argument('--retry', action='store_true', default=False, help='retry missing data')
    args = parser.parse_args()

    if args.retry:
        retry()
    else:
        blast('AGAAUACACCGCCUUGAUAUG')
