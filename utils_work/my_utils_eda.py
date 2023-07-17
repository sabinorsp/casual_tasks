def load_data_csv_by_file(path: str, sep: str, encoding: str) -> dict:
    """
    Loads all CSV files from a directory and returns a dictionary of dataframes.

    Parameters:
        path (str): The path of the directory containing the CSV files.
        sep (str): The separator character used in the CSV files.
        encoding (str): The encoding of the CSV files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where each key is the name of a CSV file (without the '.csv' extension)
                                  and each value is a pandas dataframe containing the data from the corresponding CSV file.
    """
    df_dic = {}
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            df_dic[filename.replace('.csv','')] = pd.read_csv(file_path, sep=sep, encoding=encoding)
    return df_dic


def summary_dataframes(df) -> print :
    """ 
    Return a summary about total register, type of columns, values null and 
    rows duplicated.
    
    Args:
        df (pandas.DataFrame) : Data
    """
    duplicated = df.duplicated().sum()
    total_rows = df.shape[0]
    print(f'Quantidade total de registros:{total_rows}. \n\n')
    print(df.info())
    print(f'\n\n Total Valores nulos:\n {df.isnull().sum()} \n\n')
    print(f'Total valores duplicados: \n {duplicated} ->  {round(duplicated/total_rows*100,2)} %')


def plot_balanced_class(series):
    """
    Plot the counts and percentages of a balanced class series using a bar chart.

    Args:
        series (pandas.Series): A pandas Series object representing a balanced class.

    Returns:
        None.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> np.random.seed(42)
        >>> series = pd.Series(np.random.choice(['A', 'B', 'C'], size=100))
        >>> plot_balanced_class(series)
    """
    value_counts = series.value_counts()
    percentages = value_counts / len(series) * 100
    colors = sns.color_palette('colorblind')
    plt.bar(value_counts.index, value_counts.values, color=colors)
    
    # add percentage labels to each bar
    for i, count in enumerate(value_counts):
        percentage = round(percentages[i], 2)
        plt.text(i, count + 0.5, f'{percentage}%', ha='center', fontsize=10)

    # set the title and axis labels
    plt.title('Balanced Class Plot to:'+ series.name)
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.show()


def reverse_dict(dic:dict) -> dict:
    ''' 
        Arg: dic = {key:value}, value is type list;
        Return a new dictionary with the reverse original key:value;
    '''
    return {val: key for key,value in dic.items() for val in value}