import numpy as np
from predictor import predict_category, get_model

# 126 features expected (random numbers between 0 and 1)
test_features = np.random.rand(126)

print("Test with Random Noise 1:")
predict_category(test_features, "basic7")
predict_category(test_features, "numbers")

print("\nTest with Zeros:")
predict_category(np.zeros(126), "basic7")
predict_category(np.zeros(126), "numbers")

print("\nTest with Random Noise 2:")
predict_category(np.random.rand(126), "basic7")
predict_category(np.random.rand(126), "numbers")
