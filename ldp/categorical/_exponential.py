# -*- coding: utf-8 -*-

# Copyright 2026 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Exponential mechanism for local DP."""

import numpy as np
import pandas as pd
import typing


def dp_exponential(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    new_column=False,
) -> pd.DataFrame:
    """Apply the Exponential mechanism to a categorical column of a dataframe.

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied.
    :type columm: string

    :param epsilon: privacy budget.
    :type epsilon: float

    :param new_column: boolean, default to False. If False, the new values obtained
        with the mechanims applied are stored in the same column. If True, a new column
        "dp_{column}" is created with the new values.
    :type  new_column: boolean

    :return: dataframe with the column transformed applying the mechanism.
    :rtype: pandas dataframe.
    """

    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    if isinstance(df[column].values[0], str) == False:
        raise ValueError(
            "Type of the column not allowed for the Exponential mechanism."
        )

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    categories = np.unique(df[column].values)

    dp_column = []
    for i in range(len(df)):
        original_value = df[column].iloc[i]
        dp_value = _probability_exp(original_value, categories, epsilon)
        dp_column.append(dp_value)

    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df


def dp_exponential_array(
    data: typing.Union[typing.List, np.ndarray],
    epsilon: float,
) -> np.ndarray:
    """Apply the Exponential mechanism to an array with categorical values.

    :param data: dataset with the data under study.
    :type data: list or numpy array

    :param epsilon: privacy budget.
    :type epsilon: float

    :return: array with data transformed applying the mechanism.
    :rtype: numpy array.
    """

    if isinstance(data[0], str) == False:
        raise ValueError(
            "Type of the column not allowed for the Exponential mechanism."
        )

    if isinstance(data, list) == True:
        data = np.array(data)

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    categories = np.unique(data)

    dp_array = []
    for original_value in data:
        dp_value = _probability_exp(original_value, categories, epsilon)
        dp_array.append(dp_value)

    return np.array(dp_array)


def _probability_exp(value, categories, epsilon):
    """
    Probability of the output of the Exponential mechanism.

    :param value: current value
    :type value: str

    :param categories: possible values of the data
    :type categories: list of strings

    :param epsilon: privacy budget.
    :type epsilon: float
    """
    sensitivity = 1
    scores = np.array([1 if value == c else 0 for c in categories])
    exp_scores = np.exp((epsilon * scores) / 2 * sensitivity)
    probs = exp_scores / sum(exp_scores)
    return np.random.choice(categories, p=probs)
