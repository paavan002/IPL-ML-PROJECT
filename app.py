from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Initialize Flask application
app = Flask(__name__)

app.static_folder = 'static'

# Load and preprocess data
df = pd.read_csv('IPL_Matches_2008_2022.csv')

df['WinningTeam'] = df['WinningTeam'].fillna('Draw')
df['Player_of_Match'] = df['Player_of_Match'].fillna('0')

df = df.drop(columns=['method', 'Margin', 'City', 'SuperOver'])

df['Team1'] = df['Team1'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
df['Team2'] = df['Team2'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
df['WinningTeam'] = df['WinningTeam'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
df['TossWinner'] = df['TossWinner'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})

X = df[['Team1', 'Team2', 'TossWinner', 'TossDecision']]
y = df['WinningTeam']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('Team1', OneHotEncoder(handle_unknown='ignore'), ['Team1']),
        ('Team2', OneHotEncoder(handle_unknown='ignore'), ['Team2']),
        ('TossWinner', OneHotEncoder(handle_unknown='ignore'), ['TossWinner']),
        ('TossDecision', OneHotEncoder(), ['TossDecision'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = request.form.to_dict()
    upcoming_match = pd.DataFrame([input_data])

    # Preprocess input data using pipeline
    upcoming_match_transformed = pipeline.named_steps['preprocessor'].transform(upcoming_match)

    # Predict using pipeline
    predicted_winner = pipeline.named_steps['classifier'].predict(upcoming_match_transformed)

    return render_template('result.html', prediction=predicted_winner[0])

if __name__ == '__main__':
    app.run(debug=True)