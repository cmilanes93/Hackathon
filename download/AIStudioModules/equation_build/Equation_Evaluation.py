def equation_evaluation(dataset,parameters):
    columnheader = parameters["column_name"]
    expression = parameters["expression"]

    dataset[columnheader] = ""

    temp_dataset_columns = dataset.columns
    dataset.columns = dataset.columns.map(lambda x: x.replace(' ', ''))
    dataset.columns = dataset.columns.map(lambda x: x.replace('-', ''))

    original_columns = list(temp_dataset_columns)
    for oldcolumn in original_columns:
        if oldcolumn in expression:
            newcolumn = oldcolumn.replace(' ','').replace('-', '')
            expression = expression.replace(oldcolumn, newcolumn)
    
    columnheader = columnheader.replace(' ', '').replace('-', '')
    return expression,columnheader