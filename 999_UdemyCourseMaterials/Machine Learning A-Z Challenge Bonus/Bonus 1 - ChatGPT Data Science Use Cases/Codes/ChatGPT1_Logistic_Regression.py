# Logistic Regression

# Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read in the data and split it into X and y
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate and print the accuracy
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
