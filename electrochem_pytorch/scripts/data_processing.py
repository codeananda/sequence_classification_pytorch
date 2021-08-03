import numpy as np
from pathlib import Path

DATA_DIR = Path('data')


def transform_to_longform_df(df):
    """
    Transform a wide-form DataFrame (df) from main.xlsx into a long-form
    one with unique columns and a RangeIndex.

    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame read in from main.xlsx. It must have columns 'Name',
        'Analyte' and 'Concentration'.

    Returns
    -------
    pandas DataFrame
        Long-form DataFrame with each column representing an analyte sample
        and having a unique name. There is a 'voltage' column with the same
        values as in main.xlsx and a RangeIndex.
    """
    # Replace spaces with underscores in Concentration column
    df['Concentration'] = df['Concentration'].str.replace(' ', '_')
    # Create new column (we will use this to extract unique names later on)
    df['metal_concentration'] = df['Analyte'] + '_' + df['Concentration']
    df = df.drop(columns=['Name', 'Analyte', 'Concentration'])
    # Transpose df (now columns are a range - 0, 1, 2, etc.)
    df_T = df.T
    # Select row containing the non-unique col names
    cols_non_unique = df_T.loc['metal_concentration', :].values
    # Create unique column names by adding an int to the end of each name
    unique_cols = [f'{col}_{i}' for i, col in enumerate(cols_non_unique)]
    df_T.columns = unique_cols
    # Drop row with index 'metal_concentration'
    df_T = df_T.drop(index='metal_concentration')

    # Create column 'voltage' (1 to -1 and back to 1 with 1002 steps)
    df_T['voltage'] = get_voltage_series()

    # Change col order so 'voltage' is at the front
    volt_col_first = np.roll(df_T.columns, 1)
    df_T = df_T.loc[:, volt_col_first]

    # Create a RangeIndex and drop old one
    df_T = df_T.reset_index()
    df_T = df_T.drop('index', axis=1)

    return df_T

def get_voltage_series():
    """
    Create a 'voltage series' numpy array: it has len 1002 and contains the
    values from 1 to -1 and back again with 0.004 between each value.

    Detailed Description
    -------
    The voltage series was defined by the client in his experiment.
    It runs from 1 to -1 and then from -1 to 1.
    The difference between each voltage is 0.004.
    The series runs along the top of the main.xlsx file and this
    is a convenience function to generate it.

    Returns
    -------
    NumPy array
        A 'voltage series' (see above for description)
    """
    # First pass from 1 to -1
    first_pass = np.arange(1, -1.004, -0.004)
    # Set position 250 to 0 (it is -8.8e-16 if we don't)
    first_pass[250] = 0
    # Repeat for second pass
    second_pass = np.arange(-1, 1.004, 0.004)
    second_pass[250] = 0

    voltage_series = np.append(first_pass, second_pass)
    return voltage_series