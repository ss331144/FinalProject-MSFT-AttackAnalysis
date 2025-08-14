import os
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import warnings

from wordcloud import WordCloud

warnings.filterwarnings("ignore")


excel_path = '/Users/shryqb/PycharmProjects/new_project_original/file_1/data/Merged_Bulletin_Data.xlsx'
df = pd.read_excel(excel_path)


# 3D PLOT
def encode_columns(df):
    """
    Converts categorical columns in the dataframe to numbers.

    df: DataFrame
    Returns the DataFrame with encoded columns.
    """
    df_encoded = df.copy()  # Create a copy of the DataFrame
    encoders = {}
    encoding_map = {}
    words_encode = {}

    # Encode categorical columns
    for col in df_encoded.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col])
        encoders[col] = encoder

        # Create a unique mapping dictionary for this column's values
        mapping = {label: int(i) for i, label in enumerate(encoder.classes_)}
        encoding_map[col] = mapping
        help_encode = []
        for val, code in mapping.items():
            help_encode.append(f"  '{val}'  ⇒  {code}")
        words_encode[col] = help_encode


    print("✅ Encoded categorical columns.")
    return df_encoded, encoder, words_encode


def generate_all_3d_plots(df):
    os.makedirs('plot_active', exist_ok=True)
    """
    Generates 3D plots for all combinations of three columns from a DataFrame.

    df: DataFrame containing the data.
    """
    # Encode columns before plotting
    df_encoded, encoders, words_encode = encode_columns(df)

    # Check if there are at least 3 columns
    if len(df_encoded.columns) < 3:
        raise ValueError("At least 3 columns are required for a 3D plot.")

    # Create all possible combinations of 3 columns
    col_combinations = combinations(df_encoded.columns, 3)

    for k, val in words_encode.items():
        print(f'encode for {k} : ')
        for i in val:
            print(i)
    # For each column combination
    for col1, col2, col3 in col_combinations:
        try:
            # Create 3D plot
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Data for each column
            x = df_encoded[col1]
            y = df_encoded[col2]
            z = df_encoded[col3]

            # Plot data in 3D
            ax.scatter(x, y, z, c=z, cmap='hot')

            # Axis labels
            ax.set_xlabel(f'{col1},  - x label')
            ax.set_ylabel(f'{col2},  - y label')
            ax.set_zlabel(f'{col3},  - z label' )

            # Save the plot to folder
            plt.savefig(os.path.join('plot_active', f'{col1}_{col2}_{col3}.png'))
            plt.show()

            print(f"✅ 3D plot for {col1}, {col2}, {col3} created successfully.")

        except Exception as e:
            print(f"⚠️ Could not create plot for {col1}, {col2}, {col3}. Error: {e}")
            continue  # Continue with next combination if there's an error



# Generate 3D plots for all combinations of 3 columns
generate_all_3d_plots(df[['Severity', 'Severity.1', 'Reboot']])


# Chi-Squared Plots
def plot_chi_square_bar(results_df, top_n=10):
    """
    Plots a bar chart of Chi-Square values for the strongest associated variables.

    Args:
        results_df (pd.DataFrame): DataFrame with Feature_1, Feature_2, and Chi2 values
        top_n (int): Number of top strongest pairs to display

    Returns:
        None
    """
    # Sort by highest Chi2 value
    sorted_results = results_df.sort_values(by="Chi2", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sorted_results,
        x="Chi2",
        y="Feature_1" + " - " + sorted_results["Feature_2"],
        hue="Feature_1" + " - " + sorted_results["Feature_2"],  # Assign y to hue
        palette="coolwarm",
        legend=False  # Disable the legend
    )

    plt.xlabel("Chi-Square Value")
    plt.ylabel("Feature Pairs")
    plt.title(f"Top {top_n} Strongest Chi-Square Associations")
    plt.show()

def chi_square_test(df, target_feature=None):
    """
    Computes Chi-Square for all pairs of categorical variables in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with categorical variables.
        target_feature (str, optional): Target variable to test against all others (if not provided, all pairs are tested).

    Returns:
        pd.DataFrame: Chi-Square results for each pair of columns.
    """
    # Filter only categorical variables
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    results = []

    # If target feature is defined, test only against it
    if target_feature and target_feature in cat_columns:
        columns_to_check = [col for col in cat_columns if col != target_feature]
        pairs = [(target_feature, col) for col in columns_to_check]
    else:
        # Test all possible pairs
        pairs = [(col1, col2) for i, col1 in enumerate(cat_columns) for col2 in cat_columns[i + 1:]]

    # Compute Chi-Square for each pair
    for col1, col2 in pairs:
        contingency_table = pd.crosstab(df[col1], df[col2])  # Frequency table
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)  # Chi-Square computation
        results.append({'Feature_1': col1, 'Feature_2': col2, 'Chi2': chi2, 'p-value': p})

    return pd.DataFrame(results)


df_res = chi_square_test(df, 'Severity')
print('chi test result : ')
print(df_res)

try:
    plot_chi_square_bar(df_res)
except Exception as e:
    print(e)


# Box Plot
def box_plots(df: pd.DataFrame, target: str):
    """
    Performs ANOVA test for each numeric variable in df against a categorical target variable
    and displays boxplot + barplot.

    :param df: DataFrame with numeric and categorical variables.
    :param target: Name of the categorical target variable.
    """

    # Check if target variable exists
    if target not in df.columns:
        raise ValueError(f"The variable {target} is not found in the DataFrame")

    # Check if target variable is categorical
    if df[target].dtype not in ['object', 'category']:
        raise ValueError("Target variable must be categorical")

    numeric_cols = df.select_dtypes(include=['number']).columns  # Identify numeric variables
    results = {}

    for col in numeric_cols:
        unique_values = df[target].nunique()
        if unique_values < 2:
            print(f"Skipping {col}: target variable contains only one category")
            continue

        groups = [df[col][df[target] == cat].dropna() for cat in df[target].unique()]
        stat, p_value = stats.f_oneway(*groups)
        results[col] = {'F-Statistic': stat, 'p-value': p_value}

        # Create Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f'Boxplot of {col} by {target}')
        plt.xlabel(target)
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.show()
box_plots(df=df, target='Severity')


# Pairwise Plot
def plot_pairwise_relationships(df, target):
    """
    Creates a pairplot to visualize pairwise relationships between numeric columns,
    encoding the target column if it's categorical.

    Args:
        df (pd.DataFrame): Input dataframe.
        target (str): Name of the target column to encode.
    """
    df_copy = df.copy()

    # Encode target column if categorical
    if df_copy[target].dtype == 'object' or str(df_copy[target].dtype).startswith('category'):
        le = LabelEncoder()
        df_copy[target] = le.fit_transform(df_copy[target])
        print(f"✅ Encoded target column '{target}' with mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Select numeric columns only
    numeric_df = df_copy.select_dtypes(include=['number'])

    # Create pairplot
    sns.pairplot(numeric_df)
    plt.suptitle('Pairwise Relationships', y=0.96)
    plt.tight_layout()

    plt.show()
    plt.close()


plot_pairwise_relationships(df, 'Severity')

# Correlation Plot
def plot_correlation_matrix_with_all_columns(df, target):
    """
    Generates a heatmap of the correlation matrix for all columns in the dataframe.
    Non-numeric columns are encoded numerically before calculating correlations.

    Args:
        df (pd.DataFrame): Input dataframe.
    """
    df_encoded = df.copy()

    # Convert non-numeric columns to numeric
    for col in df_encoded.select_dtypes(exclude='number').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    plt.figure(figsize=(10, 8))
    correlation_matrix = df_encoded.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (with Encoded Non-Numeric Columns)')
    plt.show()
    plt.close()
    print("✅ Correlation matrix plot saved successfully.")
    return correlation_matrix[target]
plot_correlation_matrix_with_all_columns(df, 'Severity')

# Word Cloud Plot
def create_wordcloud_for_target(df, target_column):
    # Ensure the column exists in DataFrame
    if target_column not in df.columns:
        print(f"Column {target_column} does not exist in the DataFrame.")
        return

    # Convert column values to text
    text = ' '.join(df[target_column].dropna().astype(str))

    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word cloud for feature - {target_column}", fontsize=30)
    plt.tight_layout(pad=0)
    plt.show()
    print(f'🎯 Cloud for {target_column} completed!')

nums = 0
create_wordcloud_for_target(df, 'Severity')
for i in df.columns:
    nums += 1
    if nums == 10:
        break
    try:
        create_wordcloud_for_target(df, i)
    except Exception as e:
        print(e)

# Dashboard X Plots
c = ['Date\nPosted', 'Bulletin\nId', 'Bulletin KB', 'Severity', 'Impact',
     'Title', 'Affected Product', 'Component KB', 'Affected Component',
     'Impact.1', 'Severity.1', 'Supersedes', 'Reboot', 'CVEs', 'Date Posted',
     'Bulletin Id']

df_fig = df.copy()
df_fig = df_fig.dropna(how='all')

dates = list(df_fig['Date\nPosted'])
severity = df_fig['Severity'].astype(str).tolist()
reboot = df_fig['Reboot'].astype(str).tolist()
impact = list(df_fig['Impact'])
encoder = LabelEncoder()
severity_encode = encoder.fit_transform(df_fig['Severity'])

print(df.dtypes)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with 2 rows and 2 columns
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scatter","colspan":2}, None],
           [{"type": "bar"}, {"type": "pie"}]],
    subplot_titles=['time by severity', 'scatter of impact', 'pie of reboot'],
)

# Scatter Plot
fig.add_trace(go.Scatter(
    x=dates, y=severity,
    mode='lines+markers', name='Severity by Dates',
    marker=dict(color='black'),
    line=dict(color='skyblue'),
), row=1, col=1)

# Bar Plot
impact_counts = df['Impact'].value_counts().sort_index()

fig.add_trace(go.Bar(
    x=impact_counts.index,
    y=impact_counts.values,
    text=[f"{val}" for val in impact_counts.values],
    textposition='outside',
    textfont=dict(size=10, color='black'),
    marker=dict(color='yellow', line=dict(color='black', width=1)),
    opacity=0.9,
    name='count of Impact'
), row=2, col=1)

# Pie Chart
rc = df['Reboot'].value_counts().sort_index()

fig.add_trace(go.Pie(
    labels=rc.index,
    values=rc.values,
    marker=dict(colors=['blue', 'red', 'green']),
    textinfo='label+percent',
    hoverinfo='label+value'
), row=2, col=2)

# Update layout
fig.update_layout(
    height=850,
    title_text="Security Microsoft Graphs",
    uniformtext_minsize=5,
    uniformtext_mode='hide'
)

os.makedirs('DashBoards', exist_ok=True)
fig.write_html("DashBoards/security_graphs.html")

fig.show()
