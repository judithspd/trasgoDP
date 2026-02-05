import unittest
from trasgodp import numerical, categorical
import numpy as np
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

    def test_error_column_rr_binary(self):
        column = "educatin"
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_error_type_exponential(self):
        epsilon = 1
        with self.assertRaises(ValueError):
            categorical.dp_exponential_array(self.data["age"].values, epsilon)

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

    def test_error_epsilon_exponential_array(self):
        epsilon = -1
        with self.assertRaises(ValueError):
            categorical.dp_exponential_array(self.data["education"].values, epsilon)

    def test_error_epsilon_rr_binary(self):
        epsilon = -1
        column = "sex"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_output_laplace(self):
        epsilon = 1
        column = "age"
        data_dp = numerical.dp_clip_laplace(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_gaussian(self):
        epsilon = 1
        delta = 1e-5
        column = "age"
        data_dp = numerical.dp_clip_gaussian(self.data, column, epsilon, delta)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_exponential(self):
        epsilon = 1
        column = "education"
        data_dp = categorical.dp_exponential(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_output_exponential_array(self):
        epsilon = 1
        data_dp = categorical.dp_exponential_array(
            self.data["education"].values, epsilon
        )
        assert isinstance(data_dp, np.ndarray)

    def test_output_rr_binary(self):
        epsilon = 1
        column = "sex"
        data_dp = categorical.dp_randomized_response_binary(self.data, column, epsilon)
        assert isinstance(data_dp, pd.DataFrame)

    def test_binary_rr_binary(self):
        epsilon = 1
        column = "workclass"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(self.data, column, epsilon)

    def test_label_rr_binary(self):
        epsilon = 1
        column = "sex"
        positive_label = "mujer"
        with self.assertRaises(ValueError):
            categorical.dp_randomized_response_binary(
                self.data, column, epsilon, positive_label=positive_label
            )


if __name__ == "__main__":
    unittest.main()
