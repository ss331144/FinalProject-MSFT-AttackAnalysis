from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from catboost import CatBoostClassifier
import os
from datetime import datetime
app = Flask(__name__)

# --- טען את המודל ---
model = CatBoostClassifier()
model.load_model("assets.cbm")

# --- טען את הדאטה לצורך מילוי ערכים לטופס ---
data_path = '/Users/shryqb/PycharmProjects/My_help_library/Project_data/Original all not null with target.xlsx'
df = pd.read_excel(data_path)

# --- תכונות שהמודל צריך ---
features = [
    'Impact', 'Title', 'Severity.1', 'Supersedes',
    'Reboot', 'CVEs', 'Affected Component', 'Component KB',
]

# --- יצירת מילון של אפשרויות לבחירה עבור כל תכונה ---
options_dict = {
    feature: sorted(df[feature].dropna().astype(str).unique())
    for feature in features
}

# --- דף פתיחה שמפנה למסך הראשי ---
@app.route('/')
def root_redirect():
    return redirect(url_for('start'))

# --- דף פתיחה (כפתור למעבר לחיזוי) ---
@app.route('/start')
def start():
    return render_template("start.html")

# --- מסך חיזוי עם טופס ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        # קבלת קלט מהטופס
        input_data = [request.form.get(f) for f in features]
        sample = [input_data]

        # חיזוי
        pred = model.predict(sample)
        prediction = f"🔐 Prediction: {pred[0]}"

        # שמירת החיזוי לקובץ CSV
        save_path = "predictions_log.csv"
        now = datetime.now()
        row_data = {
            'Date': now.strftime('%Y-%m-%d'),
            'Time': now.strftime('%H:%M:%S'),
        }
        row_data.update({f: v for f, v in zip(features, input_data)})
        row_data['Prediction'] = pred[0]

        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            df_existing = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)
        else:
            df_existing = pd.DataFrame([row_data])

        df_existing.to_csv(save_path, index=False)

    # שליחה לדף HTML עם תוצאה
    return render_template(
        "index.html",
        features=features,
        options_dict=options_dict,
        prediction=prediction
    )

# --- הרצת האפליקציה ---
if __name__ == '__main__':
    app.run(debug=True)
