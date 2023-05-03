import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import pickle

# Load the dataset
df = pd.read_csv('pharmacy_data.csv')

# Clean the data
df = df.dropna()
df = df.drop(['CustomerID', 'Country'], axis=1)
df = pd.get_dummies(df, columns=['Gender', 'AgeGroup', 'ProductCategory'])

# Split the data into training and testing sets
X = df.drop(['Purchase'], axis=1)
y = df['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['No Purchase', 'Purchase'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# Save the trained model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save the output to a file
with open('decision_tree_output.txt', 'w') as f:
    f.write('Accuracy: {}\n'.format(accuracy))
    f.write('Decision Tree:\n')
    f.write(dot_data)