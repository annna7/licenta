import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler


def downsample_data(data, labels, target_length):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    num_samples, num_features = data.shape

    if num_samples < target_length:
        raise ValueError(
            f"Cannot downsample to {target_length}; only have {num_samples} samples available."
        )

    downsample_factor = num_samples // target_length
    usable = downsample_factor * target_length
    truncated_data = data[:usable]
    truncated_labels = labels[:usable]

    reshaped_data = truncated_data.reshape(target_length, downsample_factor, num_features)
    reshaped_labels = truncated_labels.reshape(target_length, downsample_factor)

    downsampled_data = np.median(reshaped_data, axis=1)
    downsampled_labels = np.max(reshaped_labels, axis=1).astype(int)

    return downsampled_data.tolist(), downsampled_labels.tolist()


def normalize_data(training_data, testing_data):
    if not isinstance(training_data, (np.ndarray, pd.DataFrame)) or not isinstance(testing_data,
                                                                                   (np.ndarray, pd.DataFrame)):
        raise TypeError("Input data should be a numpy array or a pandas dataframe.")

    normalizer = MinMaxScaler(feature_range=(0, 1))

    normalized_training_data = normalizer.fit_transform(training_data)
    normalized_testing_data = normalizer.transform(testing_data)

    return normalized_training_data, normalized_testing_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess the SWaT dataset.')
    parser.add_argument('--train_path', type=str, default='../raw_data/swat/SWaT_Dataset_Normal_v1.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test_path', type=str, default='../raw_data/swat/SWaT_Dataset_Attack_v0.csv',
                        help='Path to the testing data CSV file.')
    parser.add_argument('--output_path', type=str, default='../data/swat/',
                        help='Path to the output directory.')
    args = parser.parse_args()

    print("Reading data from CSV files...")
    train = pd.read_csv(args.train_path, low_memory=False, decimal=',', quotechar='"').iloc[:, 1:]
    test = pd.read_csv(args.test_path, low_memory=False, sep=';', decimal=',', quotechar='"').iloc[:, 1:]

    train = train.rename(columns={'Normal/Attack': 'attack'})
    test = test.rename(columns={'Normal/Attack': 'attack'})

    # replace 'Normal' with 0 and 'Attack' with 1
    train['attack'] = train['attack'].replace(['Normal', 'Attack'], [0, 1])
    test['attack'] = test['attack'].replace(['Normal', 'Attack', 'A ttack'], [0, 1, 1])

    print("Handling missing values...")
    _arr_train = train.values.astype(float)  # (num_rows, num_cols)
    col_means = np.nanmean(_arr_train, axis=0)  # shape = (num_cols,)
    nan_r, nan_c = np.where(np.isnan(_arr_train))
    _arr_train[nan_r, nan_c] = np.take(col_means, nan_c)

    allnan_cols = np.isnan(col_means)  # boolean array, length = num_cols
    if all(allnan_cols) is False:
        _arr_train[:, allnan_cols] = 0.0

    train[:] = _arr_train

    _arr_test = test.values.astype(float)
    col_means_test = np.nanmean(_arr_test, axis=0)
    nan_r_t, nan_c_t = np.where(np.isnan(_arr_test))
    _arr_test[nan_r_t, nan_c_t] = np.take(col_means_test, nan_c_t)

    allnan_cols_t = np.isnan(col_means_test)
    if all(allnan_cols_t) is False:
        _arr_test[:, allnan_cols_t] = 0.0

    test[:] = _arr_test

    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    train_labels = train['attack']
    test_labels = test['attack']
    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    print("Normalizing data...")
    x_train, x_test = normalize_data(train, test)

    for i, col in enumerate(train.columns):
        train[col] = x_train[:, i]
        test[col] = x_test[:, i]

    print("Downsampling data...")
    d_train_x, d_train_labels = downsample_data(train, train_labels, 10)
    d_test_x, d_test_labels = downsample_data(test, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)
    train_df['attack'] = d_train_labels
    test_df['attack'] = d_test_labels

    print("Dropping first 2160 rows from training data...")
    train_df = train_df.iloc[2160:]

    train_df.to_csv(os.path.join(args.output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_path, 'test.csv'), index=False)

    with open(os.path.join(args.output_path, 'list.txt'), 'w') as f:
        for col in train.columns:
            f.write(col + '\n')
    print("Done!")

    print(train_df.shape)
    print(test_df.shape)
    print(train_df.head())
    print(test_df.head())


if __name__ == "__main__":
    main()