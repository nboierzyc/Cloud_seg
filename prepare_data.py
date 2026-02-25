"""Generate train_fold.csv with MultilabelStratifiedKFold splits."""

import os

import numpy as np
import pandas as pd
import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import config

LABEL_MAP = {'Fish': 0, 'Flower': 1, 'Gravel': 2, 'Sugar': 3}


def main():
    csv_path = os.path.join(config.DATA_DIR, "train.csv")
    df = pd.read_csv(csv_path)

    # Parse Image_Label into Image, Label, LabelIndex
    df['Image'] = df['Image_Label'].map(lambda v: v[:v.find('_')])
    df['Label'] = df['Image_Label'].map(lambda v: v[v.find('_') + 1:])
    df['LabelIndex'] = df['Label'].map(lambda v: LABEL_MAP[v])

    # Build multi-label matrix per image
    X, y, image_ids = [], [], []
    df_group = df.groupby('Image')
    for i, (key, gdf) in tqdm.tqdm(enumerate(df_group), total=len(df_group)):
        X.append([i])
        ml = np.array([0, 0, 0, 0])
        gdf_notnull = gdf.dropna(subset=['EncodedPixels'])
        ml[gdf_notnull['LabelIndex'].values] = 1
        y.append(ml)
        image_ids.append(key)

    X = np.array(X)
    y = np.array(y)

    mskf = MultilabelStratifiedKFold(n_splits=config.NUM_FOLDS, random_state=1234, shuffle=True)

    df['Fold'] = -1
    df = df.set_index('Image')
    for fold_idx, (_, val_index) in enumerate(mskf.split(X, y)):
        for i in val_index:
            df.loc[image_ids[i], 'Fold'] = fold_idx

    df = df.reset_index()

    out_path = os.path.join(config.DATA_DIR, "train_fold.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Fold distribution:\n{df.groupby('Fold').Image.nunique()}")


if __name__ == "__main__":
    main()
