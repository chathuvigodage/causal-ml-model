import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split


class Data_object():
    def __init__(self, args):

        np.random.seed(args['seed'])

        self.test_fraction = args['test_fraction']
        self.val_fraction = args['val_fraction']

        loc = args['dataset'] + '.csv'
        info_loc = args['dataset'] + '_info.csv'

        # -------------------------
        # Load data
        # -------------------------
        data = pd.read_csv(loc)
        info = pd.read_csv(info_loc)

        # -------------------------
        # Filter columns
        # -------------------------
        allowed = info['Variable_Name'].tolist()
        data = data[allowed]

        # -------------------------
        # Extract dosage (D)
        # -------------------------
        dosage_col = info[info['Variable_Type'] == 'dosage']['Variable_Name'].iloc[0]
        self.d = data[dosage_col].to_numpy()
        data.drop(dosage_col, axis=1, inplace=True)

        # -------------------------
        # Extract outcome (Y)
        # -------------------------
        target_col = info[info['Variable_Type'] == 'target']['Variable_Name'].iloc[0]
        self.y = data[target_col].to_numpy()
        data.drop(target_col, axis=1, inplace=True)

        # -------------------------
        # Drop ID columns (if any)
        # -------------------------
        for col in info[info['Variable_Type'] == 'id']['Variable_Name']:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        # -------------------------
        # Dummify categorical variables
        # -------------------------
        for col in info[info['Variable_Type'] == 'cat']['Variable_Name']:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data.drop(col, axis=1), dummies], axis=1)

        # -------------------------
        # SAVE FEATURE SCHEMA (ORDER MATTERS)
        # -------------------------
        self.feature_names = list(data.columns)

        # -------------------------
        # Convert to NumPy
        # -------------------------
        self.x = data.to_numpy().astype(np.float32)

        # -------------------------
        # Normalize X and D
        # -------------------------
        # Save normalization stats BEFORE normalizing so inference can replicate
        self.x_min = self.x.min(axis=0).tolist()
        self.x_max = self.x.max(axis=0).tolist()
        self.d_min = float(self.d.min())
        self.d_max = float(self.d.max())

        self.x = self.normalize(self.x)
        self.d = self.normalize(self.d.reshape(-1, 1)).flatten()

        # Persist normalization stats alongside feature_schema.json
        norm_stats = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "d_min": self.d_min,
            "d_max": self.d_max
        }
        with open("norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

        # -------------------------
        # Train / Val / Test split
        # -------------------------
        self.dataset, self.dataset_train, self.dataset_val, self.dataset_test = \
            self.split_data()

    def normalize(self, x):
        return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-8)

    def split_data(self):
        idx = np.arange(len(self.x))

        train_idx, test_idx = train_test_split(
            idx, test_size=self.test_fraction, random_state=42
        )

        train_idx, val_idx = train_test_split(
            train_idx, test_size=self.val_fraction, random_state=42
        )

        def pack(indices):
            return {
                'x': self.x[indices],
                'd': self.d[indices],
                'y': self.y[indices],
            }

        return (
            pack(idx),
            pack(train_idx),
            pack(val_idx),
            pack(test_idx)
        )
