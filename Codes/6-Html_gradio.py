import os
from catboost import CatBoostClassifier
import dash
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
from pandas.core.internals.blocks import external_values
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder

# Read the data
'''
Parameters for my model
'''
df = pd.read_csv('/Users/shryqb/PycharmProjects/new_project_original/file_1/data/Merged_Bulletin_Data.csv')
model_path = '/Users/shryqb/PycharmProjects/new_project_original/file_1/my_catboost_model.cbm'
target = 'Severity'
features = [
    'Impact',
    'Title',
    'Severity.1',
    'Supersedes',
    'Reboot',
    'CVEs',
    'Affected Component',
    'Component KB',
]

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
MyGlobalModel = None
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# Create the application
# Load last CSS in assets file automatically
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/flat-ui/2.3.0/css/flat-ui.min.css'
                ]
)

def run_dashboard(features, save_label):
    dropdowns = []
    for i, feature in enumerate(features):
        dropdowns.append(
            html.Div([
                html.Label(feature),
                dcc.Dropdown(
                    id={"type": "dropdown", "index": i},
                    value=save_label[i][0]['value'],
                    options=save_label[i],
                    placeholder=f"choose {feature}",
                    className="form-control"
                )
            ], style={"margin-bottom": "10px",})
        )
    app.layout = html.Div([
        html.H2("Prediction Microsoft Security", className="text-center mb-4"),
        html.Div(dropdowns, className="container"),
        html.Button("Send", id="submit-button", className="btn btn-primary mt-3"),
        html.Div(id="output-container", className="mt-4"),
        html.Div(id="output-model-result", className="mt-4")
    ])

# Callback for displaying the selection
@app.callback(
    Output("output-container", "children"),
    Input("submit-button", "n_clicks"),
    State({"type": "dropdown", "index": ALL}, "value")
)
def update_output(n_clicks, selected_values):
    global MyGlobalModel
    prediction = None
    if n_clicks:
        selected = [val if val is not None else "None" for val in selected_values]
        if not selected:
            return 'All variables need to be filled'
        print(f'selected {selected}')
        # Here you call the function that runs the model
        if n_clicks and selected_values:
            # The process you want to perform with the model and selected values
            # For example: making a prediction on the selected values
            # Suppose you want to predict a result for the selected values:
            # Convert selected_values to the appropriate format (according to your model)
            # selected_values = pd.get_dummies([selected_values])
            prediction_ = MyGlobalModel.predict([selected_values])  # This is an example - update as needed
            prediction = prediction_
        if isinstance(prediction, (list, np.ndarray)):
            results = prediction[0]
        else:
            results = prediction
        return (
            # html.Div([
            #    html.H5("Selected values:"),
            #    html.Ul([html.Li(val) for val in selected]),
            #    html.Br()
            # ]),
            html.Div([
                html.H3('Model Prediction:'),
                html.H3(list(results))
            ])
        )
    return None

def runApp(df, features_, model):
    global MyGlobalModel
    save_label = []
    for feature in features_:
        unique_values = df[feature].dropna().unique()
        dropdown_options = [{"label": str(value), "value": str(value)} for value in unique_values]
        save_label.append(dropdown_options)
    MyGlobalModel = model
    run_dashboard(features=features_, save_label=save_label)
    app.run_server(debug=True, use_reloader=False)

# model, metrics, X_train = train_catboost_(df=df, iterations=156, features=features, target='Severity', Depth=6, LR=0.10593804107942982, test=0.443251)
from catboost import CatBoostClassifier

if __name__ == '__main__':
    model = CatBoostClassifier()
    model.load_model(model_path)
    runApp(df=df[features + ['Severity']], features_=features, model=model)
