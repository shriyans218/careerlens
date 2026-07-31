"""
STEP 2: Preprocess + split.

Rules being applied here:
1. Split BEFORE fitting any transformer (scaler, encoder). Fitting on the
   full dataset and THEN splitting leaks test-set statistics into training.
2. Stratify the split on the target column so each of the 90 careers is
   proportionally represented in both train and test (critical when you
   only have ~100 samples per class - random splitting could easily starve
   a class in the test set).
3. One-hot encode the categorical column (Field). Scale the numeric
   columns since Logistic Regression is sensitive to feature scale
   (tree models like RF/XGBoost don't need this, but it doesn't hurt them).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("/home/claude/careerlens/data/cpp.csv")

NUMERIC_COLS = [
    "GPA", "Extracurricular_Activities", "Internships", "Projects",
    "Leadership_Positions", "Field_Specific_Courses", "Research_Experience",
    "Coding_Skills", "Communication_Skills", "Problem_Solving_Skills",
    "Teamwork_Skills", "Analytical_Skills", "Presentation_Skills",
    "Networking_Skills", "Industry_Certifications",
]
CATEGORICAL_COLS = ["Field"]

X = df[NUMERIC_COLS + CATEGORICAL_COLS]
y_raw = df["Career"]

# Encode target labels (Career names) into integers 0..89
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Stratified 80/20 split — "stratify=y" guarantees the same class
# proportions in train and test, essential with only ~100 rows/class.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ColumnTransformer applies different preprocessing to different column
# groups, and — critically — is FIT on X_train only, then just APPLIED
# (transform, not fit_transform) to X_test.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)   # fit + transform
X_test_proc = preprocessor.transform(X_test)          # transform ONLY

print("Train shape:", X_train_proc.shape)
print("Test shape:", X_test_proc.shape)
print("Classes:", len(label_encoder.classes_))

# Save everything the next script needs
joblib.dump(
    {
        "X_train": X_train_proc, "X_test": X_test_proc,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor, "label_encoder": label_encoder,
    },
    "/home/claude/careerlens/model_out/cpp_split.joblib",
)
print("Saved split + preprocessor to cpp_split.joblib")
