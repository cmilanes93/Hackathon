import numpy as np
import pandas as pd
import scipy.stats as ss
from .dataSummary.columnType import identifyColumnType
def cramers_v(confusion_matrix, correct):
    """ calculate Cramers V statistic for categorial-categorial association."""
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    sum_confusionmatrix = confusion_matrix.sum().sum()
    phi2 = chi2 / sum_confusionmatrix
    total_rows, total_col = confusion_matrix.shape
    
    phi2corr = max(
        0, phi2 - ((total_col-1)*(total_rows-1))/(sum_confusionmatrix-1))
    rcorr = total_rows - ((total_rows-1)**2) / \
        (sum_confusionmatrix-1)
    kcorr = total_col - ((total_col-1)**2)/(sum_confusionmatrix-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        
        return np.sqrt(phi2 / min(total_col - 1, total_rows - 1))
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def correlation_ratio(categories, measurements):
    """
    Calculates the Correlation Ration (sometimes marked by the greek letter Eta) for categorical-continuous association.
    """
    measurements.fillna(0, inplace=True)
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(
            fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    
    y_total_avg = np.sum(np.multiply(
        y_avg_array, n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(
        np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(
        np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def correlation_matrix(var, vardtype_, pair_var, pair_vardtype_, dataset):
    miscelleneous = False
    
    if (pair_vardtype_ == "categorical" and dataset[pair_var].dtype == "object") and (vardtype_ == "categorical" and dataset[var].dtype == "object"):
        confusion_matrix = pd.crosstab(
            dataset[var], dataset[pair_var]).to_numpy()
        if confusion_matrix.shape[0]==2:
            correct=False
        else:
            correct=True
        corr_value = cramers_v(confusion_matrix, correct)
        if np.isnan(corr_value):
            corr_value = ''
    elif vardtype_ == "categorical" and dataset[var].dtype == "object":
        corr_value = correlation_ratio(
            dataset[var], dataset[pair_var])
        
    elif pair_vardtype_ == "categorical" and dataset[pair_var].dtype == "object" :
        corr_value = correlation_ratio(
            dataset[pair_var], dataset[var])
    
    elif vardtype_ in {"miscelleneous", "text", "Time", "Date"}:
        miscelleneous = True
        corr_value = None
    else:
        corr_value = np.corrcoef(dataset[var], dataset[pair_var])[0, 1]
        if np.isnan(corr_value):
            corr_value = None 
    
    return corr_value, miscelleneous    

def plot_correlation_matrix(dataset):
    corr_list = {}
    column_type_map = identifyColumnType(dataset)
    for col_name, col_type in column_type_map.items():
        if not col_type in ['categorical', 'numerical']:
            continue
        for pair_name, pair_type in column_type_map.items():
            if not pair_type in ['categorical', 'numerical']:
                continue
            corr_value, misc = correlation_matrix(
                col_name,
                col_type,
                pair_name,
                pair_type,
                dataset
            )

            if not misc:
                if col_name not in corr_list.keys():
                    corr_list[col_name] = []
                corr_list[col_name].append(corr_value)
    
    return pd.DataFrame(corr_list, index=list(corr_list.keys()))
