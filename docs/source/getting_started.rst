Getting started
###############

This first example uses the `adult dataset`_. The idea is to apply local DP, first, to a numerical attribute (age) and second, to a categorical one (workclass). The resulting values will be stored in two new columns of the dataframe.

.. code-block:: python

   import pandas as pd
   from ldp.numerical import dp_clip_laplace
   from ldp.categorical import dp_exponential

   # Read and process the data
   data = pd.read_csv("examples/adult.csv")
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

   # Apply DP for the attribute age:
   column_num = "age"
   epsilon1 = 10
   df_age = dp_clip_laplace(data, column_num, epsilon1, new_column=True)

   # Apply DP for the attribute workclass:
   column_cat = "workclass"
   epsilon2 = 5
   df = dp_exponential(df_age, column_cat, epsilon2, new_column=True)
   
   

.. _adult dataset: https://archive.ics.uci.edu/ml/datasets/adult

