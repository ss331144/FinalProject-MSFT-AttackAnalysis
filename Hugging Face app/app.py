import gradio as gr
import pandas as pd
from catboost import CatBoostClassifier

# טען את המודל
model = CatBoostClassifier()
model.load_model("model.cbm")

# רשימת הפיצ'רים
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

# טען את הנתונים (כדי לאסוף ערכים ייחודיים לכל פיצ'ר)
df = pd.read_csv("/Users/shryqb/PycharmProjects/new_project_original/file_1/data/Merged_Bulletin_Data.csv")  # ודא שקובץ זה נמצא בתיקייה

# צור תפריטים נפתחים עם הערכים האפשריים
dropdown_inputs = []
for feature in features:
    options = sorted(df[feature].dropna().unique().astype(str))
    dropdown_inputs.append(gr.Dropdown(choices=options, label=feature))

# פונקציית חיזוי
def predict(*inputs):
    sample = [list(inputs)]
    prediction = model.predict(sample)
    return f"🔮 Prediction: {prediction[0]}"

# הפעלת הממשק
demo = gr.Interface(
    fn=predict,
    inputs=dropdown_inputs,
    outputs="text",
    title="Microsoft Security Attack Classifier",
    description="בחר ערכים לכל שדה וראה תחזית מהמנוע של CatBoost"
)

if __name__ == "__main__":
    demo.launch()
import gradio as gr
import pandas as pd
from catboost import CatBoostClassifier

# טען את המודל
model = CatBoostClassifier()
model.load_model("model.cbm")

# רשימת הפיצ'רים
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

# טען את הנתונים (כדי לאסוף ערכים ייחודיים לכל פיצ'ר)
df = pd.read_csv("Merged_Bulletin_Data.csv")  # ודא שקובץ זה נמצא בתיקייה

# צור תפריטים נפתחים עם הערכים האפשריים
dropdown_inputs = []
for feature in features:
    options = sorted(df[feature].dropna().unique().astype(str))
    dropdown_inputs.append(gr.Dropdown(choices=options, label=feature))

# פונקציית חיזוי
def predict(*inputs):
    sample = [list(inputs)]
    prediction = model.predict(sample)
    return f"🔮 Prediction: {prediction[0]}"

# הפעלת הממשק
demo = gr.Interface(
    fn=predict,
    inputs=dropdown_inputs,
    outputs="text",
    title="Microsoft Security Attack Classifier",
    description="בחר ערכים לכל שדה וראה תחזית מהמנוע של CatBoost"
)

if __name__ == "__main__":
    demo.launch()
