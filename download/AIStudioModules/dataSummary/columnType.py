def identifyColumnType(dataset):
    #parse through columns and identify the column type
    col_types = {}
    for var in dataset.columns:
        col_types[var] = get_dtype_of_col(dataset[var])
    
    return col_types


def get_dtype_of_col(dataset_col):

    n_rows = dataset_col.shape[0]
    # check number of words to identify "text" datatype
    if dataset_col.dtype == 'object' and (1. * ((dataset_col.str.count(' ') + 1) > 5).values.sum() / dataset_col.count() > 0.2):
        return "text"
    else:
        #  check whether it is likely to be unique
        likely_unique = 1.*dataset_col.nunique() / \
            dataset_col.count() == 1
        if likely_unique:
            return "miscelleneous"
        elif dataset_col.nunique() == 1:
            return "miscelleneous"
        #  wrt rows threshold value is being defined
        elif n_rows < 200 and (1.*dataset_col.nunique() / dataset_col.count() < 0.10):
            return "categorical"
        elif n_rows >= 200 and n_rows <= 2000 and (1.*dataset_col.nunique() / dataset_col.count() < 0.05):
            return "categorical"
        elif len(str(n_rows)) == 4 and  (1.*dataset_col.nunique() / dataset_col.count() < 0.005):
            return "categorical"
        elif (1.*dataset_col.nunique() / dataset_col.count() < 0.002):
            return "categorical"
        elif dataset_col.dtype == 'object':
            return "miscelleneous"
        elif dataset_col.dtype == 'datetime64[ns]':
            return "miscelleneous"
        else:
            return "numerical"
