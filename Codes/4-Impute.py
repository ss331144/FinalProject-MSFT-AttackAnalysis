import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score

max_d = 12
fontsize = 3
dict_best_model = {}
os.makedirs('decision trees', exist_ok=True)
os.makedirs('output data', exist_ok=True)

##############################################################################################################################################
##############################################################################################################################################
'''
Fills missing values in categorical columns according to a defined strategy.
Uses SimpleImputer to fill missing values such as None or NaN.
'''
print(f'[parameters : max deeps = {max_d} , fontsize = {fontsize}]')
print('='*444)

def fill_missing_categorical_values(df, strategy='most_frequent'):
    # Convert None to nan
    df = df.replace({None: np.nan})
    # Create SimpleImputer for categorical columns
    imputer = SimpleImputer(strategy=strategy, fill_value='missing')
    # Fit the imputer to the DataFrame
    filled_array = imputer.fit_transform(df)
    df = pd.DataFrame(filled_array, columns=df.columns)
    columns = df.columns
    # Display the fill values chosen
    for col, fill_value in zip(columns, imputer.statistics_):
        print(f"column {col} filled by :  {fill_value}")
    # Return the filled DataFrame
    return df

##############################################################################################################################################
##############################################################################################################################################
'''
Splits the data into two groups: one with missing values in the target column and one without.
Allows separate handling of data with and without target values.
'''
def split_by_target_null(df):
    df_null_target = df[df[target].isnull()]
    df_not_null_target = df[df[target].notnull()]
    return df_null_target, df_not_null_target

##############################################################################################################################################
##############################################################################################################################################
'''
Trains a decision tree model to predict missing values in the target column.
Displays a confusion matrix, model accuracy, and a decision tree plot.
'''
def tree_model(df, df_null, target, name):
    df_not_target = df.drop(columns=[target])
    df_null_not_target = df_null.drop(columns=[target]) # This DataFrame has null in target
    df_combined = pd.concat([df_not_target, df_null_not_target])
    df_combined_dummies = pd.get_dummies(df_combined)
    df1_dummies = df_combined_dummies.iloc[:len(df)]
    df2_dummies_ob = df_combined_dummies.iloc[len(df):]
    X = df1_dummies
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{cm}')
    accuracy = accuracy_score(y_test, y_pred)
    # Predict for df_null
    df_2_predict = model.predict(df2_dummies_ob)
    df_null.loc[:, target] = df_2_predict
    print(f'predict :\n {df_2_predict}')
    # Limit feature names to 20 characters
    max_length = 20
    truncated_feature_names = [name[:max_length] for name in X.columns]
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,  # The decision tree model itself
        filled=True,  # Color the box according to the class (True=full color, False=no)
        feature_names=truncated_feature_names,  # Feature names (columns) used to train the model
        class_names=model.classes_.astype(str),  # Class names in labels (here converted to strings)
        rounded=True,  # Round the corners of the box (True=rounded corners, False=sharp corners)
        max_depth=max_d,  # The depth (max_depth) of the tree, i.e., up to what level to draw the tree
        fontsize=fontsize,  # Font size of the text in the tree
        precision=2,  # Precision level for displaying information inside the box (like percentages)
        proportion=True,  # Whether to display the relative values of each class (True=relative proportions)
        label='all',  # Specify whether to display all labels at each node (can also be 'root', 'none', 'all')
    )
    plt.title("Decision Tree Model")
    # plt.show()
    plt.savefig(os.path.join('decision trees', f'tree of {name} for predict missing target.png'))
    dict_best_model['decision tree model missing data'] = accuracy
    return model, accuracy, df_null

##############################################################################################################################################
##############################################################################################################################################
'''
Trains a decision tree model on the complete data and calculates accuracy.
Predicts values for new data and displays a decision tree plot.
The complete data without missing values.
'''
def train_decision_tree_filled_data(df, target, new_data):
    # Convert categorical variables to dummies
    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target])  # All columns except the target
    y = df[target]  # The column representing the target
    X = pd.get_dummies(X, drop_first=True) # Dummy features
    print(f'--------- shape of dummy : {X.shape}')
    # pd.concat([X,y]).to_csv(os.path.join('output data', 'dummies values frame.csv'))
    # print('---------- saved dummies values frame successfully')
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Create the model
    model = DecisionTreeClassifier(random_state=42)
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix for full dataFrame : \n{cm}')
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    # Limit feature names to 20 characters
    max_length = 20
    truncated_feature_names = [name[:max_length] for name in X.columns]
    # Display decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,  # The decision tree model itself
        filled=True,  # Color the box according to the class (True=full color, False=no)
        feature_names=truncated_feature_names,  # Feature names (columns) used to train the model
        class_names=model.classes_.astype(str),  # Class names in labels (here converted to strings)
        rounded=True,  # Round the corners of the box (True=rounded corners, False=sharp corners)
        max_depth=max_d,  # The depth (max_depth) of the tree, i.e., up to what level to draw the tree
        fontsize=fontsize,  # Font size of the text in the tree
        precision=2,  # Precision level for displaying information inside the box (like percentages)
        proportion=True,  # Whether to display the relative values of each class (True=relative proportions)
        label='all',  # Specify whether to display all labels at each node (can also be 'root', 'none', 'all')
    )
    plt.title("Decision Tree Model")
    plt.savefig(os.path.join('decision trees', f'decision_tree_{target} - full dataFrame.png'))  # Save the tree as an image file
    try:
        # Convert the new observation to dummies
        new_data_dummies = pd.get_dummies(new_data)
        # Ensure the new observation includes all columns of X from the model, and fill missing values with 0
        new_data_dummies = new_data_dummies.reindex(columns=X.columns, fill_value=0)
        # Predict with the model
        new_prediction = model.predict(new_data_dummies)
        # Print the result
        # print(f'Predicted value for the new data: {new_prediction}')
        dict_best_model['decision tree model filled all data'] = accuracy
    except Exception as e:
        print(e)
    print(f'evaluate model : decision tree model - acc {accuracy} , f1 {f1} , precision {precision}')
    return model, accuracy, new_prediction

##############################################################################################################################################
##############################################################################################################################################
'''
Converts a date column to a numeric format with day, month, and year.
Automatically identifies the date separator and handles conversion errors.
'''
def convert_to_datetime(df, column_name, separator='-'):
    df[column_name] = df[column_name].astype(str)
    if df[column_name].str.contains(r'\.', regex=True).any():
        separator = '.'
    elif df[column_name].str.contains(r'\\', regex=True).any():
        separator = '\\'
    else:
        separator = '-'
    try:
        # Split the date into parts (day, month, year)
        df[['Year', 'Month', 'Day']] = df[column_name].str.split(separator, expand=True)
        # Convert columns to numeric data (if not already)
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        # Delete the original date column
        df = df.drop(columns=[column_name])
    except Exception as e:
        print(f"Error occurred while converting date: {e}")
    return df

#### Running
####
df = pd.read_csv('/Users/shryqb/PycharmProjects/new_project_original/file_1/data/Merged_Bulletin_Data.csv')
data_csv = df.copy()
last_row = data_csv.iloc[-1:]  # Duplicate the last row of the DataFrame
data_csv = pd.concat([data_csv, last_row], ignore_index=True)  # Add the duplicated row as new data
data_csv.at[data_csv.index[-1], 'Severity'] = np.nan  # Change the value in the 'Severity' column of the last row to NaN
print(f'null in last row : \n{data_csv.iloc[-1]["Severity"]}')  # Print the value of 'Severity' in the last row
features =  [
    "Date Posted", "Bulletin Id", "Bulletin KB",  "Impact", "Title",
    "Affected Product", "Component KB", "Affected Component", "Impact.1",
    "Severity.1", "Supersedes", "Reboot", "CVEs"
]
target = 'Severity' # Target
target_row = data_csv[target] # Get the target
data = data_csv[features] # Get the features from the data
df_features = pd.DataFrame(data) # DataFrame of features
df_features = convert_to_datetime(df_features, 'Date Posted') # Split the date from the date format because it doesn't work
df_target = pd.DataFrame(target_row, columns=[target]) # DataFrame of target
null_target = df_target.isnull().sum() # Check how many nulls are in the target
print(f'null in target row : \n{null_target}') # Print nulls of target
print(f'null in all data before impute : \n{data_csv.isnull().sum()}') # Print nulls of target
print('-'*222)
filled_df = fill_missing_categorical_values(df_features) # Fill missing values in categorical columns
filled_df.to_excel(os.path.join('output data', '1 - Data filled in features.xlsx'), index=False) # Save
full_df_concat = pd.concat([filled_df, df_target], axis=1) # Concatenate the complete data of target and features
full_df_concat.to_excel(os.path.join('output data', r'2 - All data filled with target.xlsx'), index=False) # Save
print(f'null value in full df :\n{full_df_concat.isnull().sum()}') # Check if there are nulls in the new complete frame
print('='*222)

# Using Fill Algorithm for missing data
if (null_target > 0).all():
    # Split the data into rows with and without missing values in the target column
    null_target_row, not_null_target_row = split_by_target_null(full_df_concat)
    null_target_row.to_csv(os.path.join('output data', 'divide dataframe null in severity.csv'))
    # Train a decision tree model on the complete data and predict the missing values
    model, accuracy, null_target_row = tree_model(not_null_target_row, null_target_row, target, 'Severity')
    print('the accuracy of first model with part of df not null (tree model) : ')  # Print message about the performance of the first model
    print(f'accuracy : {accuracy}')  # Print model accuracy
    full_df_all = pd.concat([not_null_target_row, null_target_row], ignore_index=True)  # Recombine data with and without missing values after filling
else:
    print("No null values in target row, training model on full data.")  # Message if there are no missing values in the target column
    full_df_all = full_df_concat  # Use all data without splitting
print('='*222)  # Print a separator line of 222 characters
full_df_all.to_excel(os.path.join('output data', r'3 - Original alldata filled with target.xlsx'), index=False) # Save
