from lightgbm import LGBMClassifier
import joblib as jl
import numpy as np


def fit_once(x_data, y_data, n_fold):
    idx = np.arange(x_data.shape[0])
    train_idx = idx[idx % 5 != n_fold]
    val_idx = idx[idx % 5 == n_fold]

    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]

    clf = LGBMClassifier(objective='multiclass', num_leaves=15, colsample_bytree=.2, n_estimators=10000)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric='logloss', early_stopping_rounds=25)
    jl.dump(clf, f'lgb_{n_fold}.bin')


def main():
    x_data, y_data = jl.load('data/train.bin')
    for i in range(5):
        fit_once(x_data, y_data, n_fold=i)


if __name__ == '__main__':
    main()
