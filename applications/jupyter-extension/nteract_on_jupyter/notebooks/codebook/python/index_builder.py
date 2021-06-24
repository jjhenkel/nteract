import subprocess
import hashlib
import pandas
import tqdm
import gzip
import json


DATA_SET = '/data/test-1k'

target_files = sorted(list([ x.strip() for x in subprocess.run([
    'bash',
    '-c',
    "find {}/processed/python -type f -name 'debug_text.csv.gz'".format(DATA_SET)
], capture_output=True).stdout.decode().split('\n') if len(x.strip()) > 0 ]))

INDEX = {}

with open(DATA_SET + '/indices/all-files.txt', 'w') as fh:
    fh.write('\n'.join([
        x.replace('.flow/debug_text.csv.gz', '') for x in target_files
    ]) + '\n')

chunk_size = 10000
idx = 0
part_idx = 1
frames = []
for file in tqdm.tqdm(target_files):
    data = pandas.read_csv(
        file,
        compression='gzip',
        dtype=str,
        delimiter='\t',
        names=['gid', 'sr', 'sc', 'er', 'ec', 'text'],
        usecols=['text']
    )
    data['file'] = [ file.replace('.flow/debug_text.csv.gz', '') ] * len(data)
    frames.append(data)
    
    # for key in res:
    #     if key not in INDEX:
    #         INDEX[key] = []
    #     INDEX[key].append(fidx)

    # idx += 1
    # if idx >= chunk_size:
    #     with gzip.open(DATA_SET + '/indices/text-to-files/part-{}.json.gz'.format(part_idx), 'w') as fout:
    #         fout.write(json.dumps(INDEX).encode('utf-8'))   
    #     idx = 0
    #     part_idx += 1
    #     INDEX = {}

pandas.concat(frames).to_parquet(
    DATA_SET + '/indices/text-to-files.parquet', index=True
)  

# with gzip.open(DATA_SET + '/indices/text-to-files/part-{}.json.gz'.format(part_idx), 'w') as fout:
#     fout.write(json.dumps(INDEX).encode('utf-8'))
