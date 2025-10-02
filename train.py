import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

import skops.io as sio


RANDOM_STATE = 42

DTYPE = {
    'Age': int,
    'Sex': object,
    'BP': object,
    'Cholesterol': object,
    'Na_to_K': float,
    'Drug': object
}

df=pd.read_csv('data/drug200.csv', dtype = DTYPE)

X = df.drop('Drug', axis=1)
y = df['Drug']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("oe", OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

Model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, class_weight='balanced')

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", Model)
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

report_train = classification_report(y_train, clf.predict(X_train), output_dict=True)
report_test = classification_report(y_test, y_pred, output_dict=True)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1: {f1}\n")
    
    
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.savefig("results/model_results.png", dpi=120)

sio.dump(clf, "model/drug_pipeline.skops")