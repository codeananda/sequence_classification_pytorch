import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path('data')

def create_cleaned_df(df, class_label_str):
    """Transform the wide-from Dataframe (df) from main.xlsx into one with
    unique row names, values 0-1001 as the column names and a label column
    containing the class label as an int.

    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame read in from main.xlsx. It must have columns 'Name',
        'Analyte' and 'Concentration'.
    class_label_str: str (len 2)
        The class label for the dataframe. It must be two characters long and
        one of 'Cu', 'Cd', 'Pb' or 'Sw'.

    Returns
    -------
    pandas DataFrame
        Wide-form dataframe with unique row and column names and a label column.

    """
    # Replace spaces with underscores in Concentration column
    df['Concentration'] = df['Concentration'].str.replace(' ', '_')
    # Create new column (we will use this to extract unique names later on)
    df['metal_concentration'] = df['Analyte'] + '_' + df['Concentration']
    df = df.drop(columns=['Name', 'Analyte', 'Concentration'])
    # Transpose df (now columns are a range - 0, 1, 2, etc.)
    df['metal_concentration'] = [f'{name}_{i}' for i, name in enumerate(df['metal_concentration'])]
    df = df.set_index('metal_concentration')
    df.index.name = None
    df.columns = range(0, 1002)
    class_label_to_int_mapping = get_class_label_to_int_mapping()
    df['label'] = class_label_to_int_mapping[class_label_str]
    return df


def get_class_label_to_int_mapping():
    """Create mapping from str labels to int labels (PyTorch expects labels
    to be ints).

    Returns
    -------
    Dict
        Dict mapping str 2-letter labels to ints
    """
    labels_str = ['Cu', 'Cd', 'Pb', 'Sw']
    label_enc = LabelEncoder()
    labels_int = label_enc.fit_transform(labels_str)
    label_to_int_mapping = dict(zip(labels_str, labels_int))
    return label_to_int_mapping