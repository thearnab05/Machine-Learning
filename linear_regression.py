import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Generate random synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    # y = 4 + 3 * X + noise
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    print("Training Linear Regression model on synthetic data...")
    model = LinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    print(f"Learned parameters:")
    print(f"Weights: {model.weights[0]:.4f} (Expected: ~3.0)")
    print(f"Bias: {model.bias:.4f} (Expected: ~4.0)")

#pending