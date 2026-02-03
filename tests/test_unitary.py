import unittest
from ldp import numerical, categorical
import pandas as pd


class TestInvalidValues(unittest.TestCase):
    data = pd.read_csv("./examples/adult.csv")
    data.columns = data.columns.str.strip()
    cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "sex",
        "native-country",
    ]
    for col in cols:
        data[col] = data[col].str.strip()

    def test_error_column_laplace(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_error_column_gaussian(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_column_exponential(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)

    def test_error_type_column_laplace(self):
        column = "education"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_error_type_column_gaussian(self):
        column = "education"
        epsilon = 1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_type_column_exponential(self):
        column = "age"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)

    def test_error_epsilon_laplace(self):
        column = "age"
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_laplace(self.data, column, epsilon)

    def test_error_epsilon_gaussian(self):
        column = "age"
        epsilon = -1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon)

    def test_error_delta_gaussian(self):
        column = "age"
        epsilon = 1
        delta = 1.1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon, delta)

    def test_error_deltaneg_gaussian(self):
        column = "age"
        epsilon = 1
        delta = -1.1
        with self.assertRaises(ValueError):
            numerical.dp_clip_gaussian(self.data, column, epsilon, delta)

    def test_error_epsilon_exponential(self):
        column = "education"
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_exponential(self.data, column, epsilon)


if __name__ == "__main__":
    unittest.main()
