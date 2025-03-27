import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

SAMPLING_METHOD = 'undersample'  # 'oversample' or 'undersample'


def analyze_dataset_composition(dfs, file_names):
    for df, name in zip(dfs, file_names):
        total = len(df)
        formal = len(df[df['label'] == 1])
        informal = len(df[df['label'] == 0])
        print(f"{name} Dataset:")
        print(f"Total samples: {total}")
        print(f"Formal: {formal} ({formal / total * 100:.2f}%)")
        print(f"Informal: {informal} ({informal / total * 100:.2f}%)\n")


def read_tsv(file_name):
    col_names = ["score", "values", "id", "text"]
    df = pd.read_csv(f"./data/{file_name}", sep="\t", header=None, names=col_names, dtype=str, encoding="ISO-8859-1")
    df["values"] = df["values"].apply(lambda x: list(map(float, x.split(","))) if isinstance(x, str) else [])
    df["score"] = df["score"].astype(float)
    df = df[(df["score"] < -0.5) | (df["score"] > 0.5)]
    df['label'] = df['score'].apply(lambda x: 1 if x > 0 else 0)
    return df


def filter_dataset(df):
    df_filt = df[df["values"].apply(lambda x: len(x) == 5)]
    df_filt = df_filt[df_filt["values"].apply(lambda x: x[-1] - x[0] < 5)]
    df_filt = df_filt[df_filt["text"].apply(lambda x: (len(x) < 400) and (len(x) > 5))]
    return df_filt


def oversample_minority_class(df, label_column):
    """
    Oversample the minority class in a binary classification dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features and label
    label_column : str
        Name of the column containing binary labels (0 or 1)

    Returns:
    --------
    pandas.DataFrame
        Dataframe with oversampled minority class
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[label_column] = y_resampled

    return resampled_df


def undersample_majority_class(df, label_column):
    """
    Undersample the majority class in a binary classification dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features and label
    label_column : str
        Name of the column containing binary labels (0 or 1)

    Returns:
    --------
    pandas.DataFrame
        Dataframe with undersampled majority class
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]

    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[label_column] = y_resampled

    return resampled_df


def balance_dataset(df, label_column, method='oversample'):
    """
    Balance the dataset using either oversampling or undersampling.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features and label
    label_column : str
        Name of the column containing binary labels (0 or 1)
    method : str, optional (default='oversample')
        Sampling method to use. Options are 'oversample' or 'undersample'

    Returns:
    --------
    pandas.DataFrame
        Balanced dataframe
    """
    if method == 'oversample':
        return oversample_minority_class(df, label_column)
    elif method == 'undersample':
        return undersample_majority_class(df, label_column)
    else:
        raise ValueError("Method must be either 'oversample' or 'undersample'")


if __name__ == '__main__':
    file_names = [
        'answers',
        'blog',
        'news',
        'email'
    ]

    # Read and filter datasets
    dfs = [read_tsv(file_name) for file_name in file_names]
    dfs = [filter_dataset(df) for df in dfs]

    # Analyze original dataset composition
    analyze_dataset_composition(dfs, file_names)

    # Balance datasets using selected method
    dfs = [balance_dataset(df, 'label', method=SAMPLING_METHOD) for df in dfs]

    # Combine and save the dataset
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")
    df.to_csv("data/dataset.csv", index=False)
