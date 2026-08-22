from http.server import BaseHTTPRequestHandler
import json
import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Load the same dataset used by the original ML project
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "student_performance_pass_fail.csv"
)


# ---------------------------------------------------------
# Train the model
# ---------------------------------------------------------

def train_model():

    df = pd.read_csv(DATA_FILE)

    # Numerical columns
    numeric_cols = [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Score"
    ]

    # Fill missing numerical values
    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # Fill missing categorical values
    df["Project_Requirements"] = (
        df["Project_Requirements"]
        .fillna(df["Project_Requirements"].mode()[0])
    )

    # Select features and target
    features = [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Score",
        "Project_Requirements"
    ]

    target = "Result"

    X = df[features]
    y = df[target]

    # Same encoding used in your notebook
    X = pd.get_dummies(
        X,
        columns=["Project_Requirements"],
        drop_first=True,
        dtype=int
    )

    # Same train-test split as your notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Same Logistic Regression pipeline
    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    return model, X.columns


# ---------------------------------------------------------
# Train once when the serverless function starts
# ---------------------------------------------------------

MODEL, MODEL_COLUMNS = train_model()


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def make_prediction(data):

    required_fields = [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Score",
        "Project_Requirements"
    ]

    # Check required fields
    for field in required_fields:

        if field not in data:
            raise ValueError(
                f"Missing field: {field}"
            )

    # Convert numeric values
    study_hours = float(data["Study_Hours"])
    attendance = float(data["Attendance"])
    previous_marks = float(data["Previous_Marks"])
    assignment_score = float(data["Assignment_Score"])

    project_requirements = str(
        data["Project_Requirements"]
    )

    # Validate values
    if study_hours < 0 or study_hours > 24:
        raise ValueError(
            "Study Hours must be between 0 and 24."
        )

    if attendance < 0 or attendance > 100:
        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    if previous_marks < 0 or previous_marks > 100:
        raise ValueError(
            "Previous Marks must be between 0 and 100."
        )

    if assignment_score < 0 or assignment_score > 100:
        raise ValueError(
            "Assignment Score must be between 0 and 100."
        )

    if project_requirements not in [
        "Met",
        "Not Met"
    ]:
        raise ValueError(
            "Project Requirements must be Met or Not Met."
        )

    # Create dataframe
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment_Score": [assignment_score],
        "Project_Requirements": [
            project_requirements
        ]
    })

    # Same encoding as notebook
    encoded = pd.get_dummies(
        new_student,
        columns=["Project_Requirements"],
        drop_first=True,
        dtype=int
    )

    # Ensure exact same columns as training data
    encoded = encoded.reindex(
        columns=MODEL_COLUMNS,
        fill_value=0
    )

    # Prediction
    prediction = MODEL.predict(encoded)[0]

    # Prediction probability
    probabilities = MODEL.predict_proba(encoded)[0]

    confidence = float(
        max(probabilities) * 100
    )

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 1)
    }


# ---------------------------------------------------------
# Vercel HTTP Handler
# ---------------------------------------------------------

class handler(BaseHTTPRequestHandler):

    def send_json(self, status_code, data):

        response = json.dumps(data)

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()

        self.wfile.write(
            response.encode("utf-8")
        )

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    def do_GET(self):

        self.send_json(
            200,
            {
                "status": "success",
                "message":
                    "Student Performance Prediction API is running."
            }
        )

    # -----------------------------------------------------
    # OPTIONS
    # -----------------------------------------------------

    def do_OPTIONS(self):

        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()

    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------

    def do_POST(self):

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            data = json.loads(
                body.decode("utf-8")
            )

            result = make_prediction(data)

            self.send_json(
                200,
                result
            )

        except json.JSONDecodeError:

            self.send_json(
                400,
                {
                    "error":
                        "Invalid JSON request."
                }
            )

        except ValueError as error:

            self.send_json(
                400,
                {
                    "error": str(error)
                }
            )

        except Exception as error:

            print(
                "Server error:",
                repr(error)
            )

            self.send_json(
                500,
                {
                    "error":
                        "Internal server error. "
                        "Please check the Vercel logs."
                }
            )