from tabpfn import TabPFNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0, stratify=y)

clf = TabPFNClassifier(device='cpu')  # 'cpu' for your current setup; change to 'gpu' if you have jax GPU
clf.fit(X_train, y_train)
print("preds:", clf.predict(X_test))
print("probs[0]:", clf.predict_proba(X_test)[0])