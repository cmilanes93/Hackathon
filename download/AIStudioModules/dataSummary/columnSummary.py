import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .columnType import get_dtype_of_col

def columnValuesDistribution(column_data):
    column_type = get_dtype_of_col(column_data)
    if column_type not in ['categorical', 'numerical']:
        print("Can not provide visuals for this type of data")
    else:
        plt.figure(figsize=(10,5))
        if column_type == 'categorical':
            data_count = column_data.value_counts()
            sns.barplot(data_count.index, data_count.values)
        else:
            sns.distplot(column_data, kde=False)
        plt.title("Statistical Summary")
        plt.show()
            