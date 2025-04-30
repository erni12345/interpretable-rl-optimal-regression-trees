import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pystreed import STreeDPiecewiseLinearRegressor

# Load dataset
df = pd.read_csv("data/auto-mpg.csv")

# Handle missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert data types
df['horsepower'] = df['horsepower'].astype(float)

# Define features and target
X = df.drop(columns=['mpg', 'car name'])
y = df['mpg']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SRT-SL model
model = STreeDPiecewiseLinearRegressor(
    max_depth=3,
    simple=True,  
    cost_complexity=0.01,
    ridge_penalty=0.1
)

# Fit the model
model.fit(X_train.values, y_train.values)

# Predict on test set
y_pred = model.predict(X_test.values)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"R² score on test set: {r2:.3f}")

# Export the tree structure
model.export_dot("srt_sl_tree.dot")
