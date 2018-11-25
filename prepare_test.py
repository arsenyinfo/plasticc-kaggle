import joblib as jl
import pandas as pd
from tqdm import tqdm
import numpy as np


def main():
    transformer = jl.load('preprocess.bin')
    path = 'data/processed_test.csv'
    chunk_size = 10000
    total = 3492889

    def _process_chunk(chunk, i):
        chunk = pd.DataFrame(chunk)
        ids = chunk.pop('object_id').apply(float).values.astype('uint16')
        chunk = chunk.values.astype('float32')
        chunk[np.isnan(chunk)] = 0
        chunk[np.isinf(chunk)] = 0
        chunk = transformer.transform(chunk)
        jl.dump((ids, chunk), f'data/chunks/test_{i}.bin')

    with open(path) as data:
        header = next(data).rstrip().split(';')

        i = 0  # chunk_id
        j = 0  # line_id
        chunk = []

        for line in tqdm(data, desc='processing data file', total=total):
            j += 1
            line = line.split(';')
            chunk.append({k: v for k, v in zip(header, line)})

            if not j % chunk_size:
                _process_chunk(chunk, i)
                i += 1
                chunk = []

        _process_chunk(chunk, i)


if __name__ == '__main__':
    main()
