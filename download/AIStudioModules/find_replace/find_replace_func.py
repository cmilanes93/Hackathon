def find_replace(replace_success, dataset, find_value, selected_feature, replaced_value, operation):

            if operation ==  "Begins with":
                if not dataset[selected_feature].astype(str).str.startswith(str(find_value)).any():
                    return False
                dataset.loc[dataset[selected_feature].astype(str).str.startswith(str(find_value)), selected_feature] = replaced_value
            elif operation == "Ends with":
                if not dataset[selected_feature].astype(str).str.endswith(str(find_value)).any():
                    return False
                dataset.loc[dataset[selected_feature].astype(str).str.endswith(str(find_value)), selected_feature] = replaced_value
            elif operation == "Contains":
                if not dataset[selected_feature].astype(str).str.contains(str(find_value)).any():
                    return False
                dataset.loc[dataset[selected_feature].astype(str).str.contains(str(find_value)), selected_feature] = replaced_value
            elif operation == "Not Begins with":
                temp = ~dataset[selected_feature].astype(str).str.startswith(str(find_value))
                if not temp.any():
                    return False
                dataset.loc[~dataset[selected_feature].astype(str).str.startswith(str(find_value)), selected_feature] = replaced_value
            elif operation == "Not Ends with":
                temp = ~dataset[selected_feature].astype(str).str.endswith(str(find_value))
                if not temp.any():
                    return False
                dataset.loc[~dataset[selected_feature].astype(str).str.endswith(str(find_value)), selected_feature] = replaced_value
            elif operation == "Not contains":
                temp = ~dataset[selected_feature].astype(str).str.contains(str(find_value))
                if not temp.any():
                    return False
                dataset.loc[~dataset[selected_feature].astype(str).str.contains(str(find_value)), selected_feature] = replaced_value
            elif operation == "Find what":
                item = r'^'+str(find_value)+'$'
                if not dataset[selected_feature].astype(str).str.match(item).any():
                    return False
                dataset.loc[dataset[selected_feature].astype(str).str.match(item), selected_feature] = replaced_value
            return True
