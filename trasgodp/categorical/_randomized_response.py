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

"""Randomized response algorithm (binary) mechanism for local DP."""

import numpy as np
import pandas as pd
import typing


def dp_randomized_response_binary(
    df: pd.DataFrame,
    column: str,
    epsilon: float,
    new_column=False,
    positive_label=None,
) -> pd.DataFrame:
    """Apply the Randomized Response mechanism to a binary column of a dataframe.

    :param df: dataframe with the data under study.
    :type df: pandas dataframe

    :param columm: column to which the DP mechanism will be applied. Binary data.
    :type columm: string

    :param epsilon: privacy budget.
    :type epsilon: float

    :param new_column: boolean, default to False. If False, the new values obtained
        with the mechanims applied are stored in the same column. If True, a new
        column 'dp_{column}' is created with the new values.
    :type  new_column: boolean

    :param positive_label: value to be assigned as 1. If None, it is assigned to
        first value.
    :type positive_label: string

    :return: dataframe with the column transformed applying the mechanism.
    :rtype: pandas dataframe.
    """
    if column not in df.keys():
        raise ValueError("Column: {column} not in the dataframe.")

    categories = np.unique(df[column].values)
    if len(categories) != 2:
        raise ValueError("Only binary attributes are supported.")

    if epsilon <= 0:
        raise ValueError("The privacy budget must be greater than 0.")

    if positive_label is None:
        positive_label = categories[0]
        negative_label = categories[1]
    else:
        if positive_label not in categories:
            raise ValueError("Positive label is not a value in the column.")
        negative_label = categories[categories != positive_label][0]

    data = df[column].values
    data_binary = [1 if v == positive_label else 0 for v in data]

    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    _dp_column = []
    for value in data_binary:
        if np.random.rand() < p:
            _dp_column.append(value)
        else:
            _dp_column.append(int(np.abs(value - 1)))

    dp_column = [positive_label if v == 1 else negative_label for v in _dp_column]

    if new_column:
        df[f"dp_{column}"] = dp_column
    else:
        df[column] = dp_column

    return df
