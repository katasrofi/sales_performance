import pandas as pd
from typing import List

# Check data
def check_data(df: pd.DataFrame) -> pd.DataFrame:
    # Print Top 5 Values
    print(f"{df.head()}\n")
    print(f"Data shape: {df.shape}")
    print(f"\nData info:\n {df.info}")
    print(f"\nData types:\n {df.dtypes}")
    print(f"\nDescribe The Data:\n {df.describe()}\n")
    print(f"Desribe All Data:\n {df.describe(include='all')}")
    print(f"\nUnique Value:")

    # Check Unique Value and Calculate the Number of Unique Value
    column_names: List[str] = list(df.columns)

    # Loop the column names
    for column in column_names:
        # Unique Values
        # unique_value = df[column].unique()
        # print(f"Count Distinct:\n {unique_value}")

        # Nunique Values
        nunique_value = df[column].nunique()
        print(f"Total of Unique Value {column}: {nunique_value}")

    # Count Null Values
    null_value: pd.Series = df.isnull().sum()
    print(f"\nTotal of Null Value: \n{null_value}")

    # Check duplicates by Column
    DuplicateByColumn: pd.DataFrame = df[df.duplicated(subset=column_names)]
    print(f"\nDuplicates By Column:\n {DuplicateByColumn}\n")


# Deletes Duplicates Value
def DropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Select the columns names
    column_names: List[str] = list(df.columns)

    # Select the duplicates
    DuplicateByColumn: pd.DataFrame = df[df.duplicated(subset=column_names)]

    # Drop the duplicates
    df_cleaned: pd.DataFrame = df[~df.duplicated(subset=column_names, keep=False)]

    return df_cleaned
