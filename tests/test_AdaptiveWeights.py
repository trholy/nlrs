import unittest
import numpy as np
from sklearn.datasets import make_regression
from nlrs.utils import AdaptiveWeights


class TestAdaptiveWeights(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        self.X, self.y = make_regression(
            n_samples=100,
            n_features=10,
            noise=0.1,
            random_state=42
        )

    def test_get_weights_lin_reg(self):
        # Test get_weights method with Linear Regression model
        model = AdaptiveWeights(model='lin_reg')
        weights = model.get_weights(self.X, self.y)
        self.assertEqual(weights.shape, (self.X.shape[1],))

    def test_get_weights_cv_ridge(self):
        # Test get_weights method with Cross-Validated Ridge Regression model
        model = AdaptiveWeights(model='cv_ridge')
        weights = model.get_weights(self.X, self.y)
        self.assertEqual(weights.shape, (self.X.shape[1],))

    def test_invalid_model(self):
        # Test if ValueError is raised for invalid model type
        with self.assertRaises(ValueError):
            model = AdaptiveWeights(model='invalid_model')
            model.get_weights(self.X, self.y)

    def test_equal_samples(self):
        # Test if ValueError is raised for unequal number of samples in X and y
        with self.assertRaises(ValueError):
            model = AdaptiveWeights(model='lin_reg')
            model.get_weights(self.X[:-1], self.y)

    def test_2d_features(self):
        # Test if ValueError is raised for 1-dimensional input feature matrix
        with self.assertRaises(ValueError):
            model = AdaptiveWeights(model='lin_reg')
            model.get_weights(np.array([1, 2, 3]), self.y)
