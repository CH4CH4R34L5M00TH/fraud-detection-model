import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load the dataset into a pandas dataframe
df = pd.read_csv('transaction_data.csv')

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('fraudulent', axis=1), df['fraudulent'], test_size=0.2, random_state=42)

# train a logistic regression model on the training set
clf = LogisticRegression()
clf.fit(X_train, y_train)

# evaluate the model on the testing set
accuracy = clf.score(X_test, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
