from sklearn.model_selection import train_test_split
import numpy as np

X = np.arange(10).reshape((10, 1))
y = np.array([1,0,1,0,1,0,1,0,1,0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
