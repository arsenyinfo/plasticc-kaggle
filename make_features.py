import pandas as pd
from tsfresh import extract_features, extract_relevant_features
from tqdm import tqdm
import joblib as jl


def stream_objects(path: str):
    with open(path) as f:
        for i, l in enumerate(f):
            pass
        total = i + 1

    with open(path) as data:
        header = next(data).split(',')

        obj_id = None
        acc = []

        for line in tqdm(data, desc='processing data file', total=total):
            line = {k: float(v) for k, v in zip(header, line.rstrip().split(','))}

            new_obj_id = line.pop('object_id')
            if new_obj_id != obj_id:
                if len(acc):
                    yield obj_id, pd.DataFrame(acc)
                acc = []
                obj_id = new_obj_id

            acc.append(line)


def make_features(chunk: pd.DataFrame) -> dict:
    chunk = chunk.fillna(method='ffill').fillna(value=0)
    features = extract_features(chunk, column_id='passband', disable_progressbar=True, n_jobs=8)

    result = {}
    for col in features.columns:
        for k, v in zip(features.index, features[col]):
            result[f'{col}_{k}'.replace('\n', '_')] = v

    return result


def merge(obj_id: int, chunk: pd.DataFrame, meta: dict):
    features = make_features(chunk)
    features.update(meta)

    features = [(k, v) for k, v in features.items()]
    features = sorted(features, key=lambda x: x[0])
    keys, values = zip(*features)
    return obj_id, keys, values


def make_lazy_features(data_path: str, metadata_path: str):
    d_meta = {}
    df_meta = pd.read_csv(metadata_path, engine='c')

    for _, row in tqdm(df_meta.iterrows(), desc='preparing meta features', total=df_meta.shape[0]):
        d = row.to_dict()
        k = int(d.pop('object_id'))
        d_meta[k] = d

    for obj_id, chunk in stream_objects(data_path):
        features = jl.delayed(merge)(obj_id=obj_id, chunk=chunk, meta=d_meta[obj_id])
        yield features


def fetch_batch_from_gen(batch_size, g):
    def _next(g_):
        try:
            return next(g_)
        except StopIteration:
            return

    return list(filter(None, [_next(g) for _ in range(batch_size)]))


def process_dataset(prefix: str, batch_size=2):
    features = make_lazy_features(data_path=f'data/{prefix}_set.csv',
                                  metadata_path=f'data/{prefix}_set_metadata.csv')
    pool = jl.Parallel(n_jobs=2, backend='sequential')

    with open(f'data/processed_{prefix}.csv', 'w') as out:
        current_keys = None
        is_finished = False

        while not is_finished:
            batch = fetch_batch_from_gen(batch_size=batch_size, g=features)
            for obj_id, keys, values in pool(batch):
                if current_keys is None:
                    keys = ['object_id', ] + list(keys)
                    line = ';'.join(keys) + '\n'
                    out.write(line)
                    current_keys = keys
                else:
                    assert tuple(current_keys[1:]) == keys, f'{tuple(current_keys[1:])[:10]}, {keys[:10]}'
                    values = [str(obj_id), ] + list(values)
                    line = ';'.join(map(str, values)) + '\n'
                    out.write(line)

            if len(batch) < batch_size:
                is_finished = True


if __name__ == '__main__':
    for prefix in ('training', 'test',):
        process_dataset(prefix)
