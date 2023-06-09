import numpy as np
import pandas as pd
import statsmodels.api as sm

# Create the data array
data = np.array([[1, 2, 3, 4],
                 [8, 9, 7, 8],
                 [5, 10, 7, 8],
                 [5, 6, 3, 8],
                 [14, 22, 7, 8],
                 [18, 12, 17, 8],
                 [14, 12, 7, 18],
                 [24, 22, 7, 8],
                 [16, 12, 27, 28],
                 [10, 32, 7, 8],
                 [14, 12, 37, 8],
                 [11, 12, 7, 38],
                 [19, 12, 57, 48],
                 [14, 12, 73, 83],
                 [22, 22, 27, 28],
                 [34, 32, 7, 38],
                 [35, 39, 37, 8],
                 [11, 6, 12, 18],
                 [15, 16, 17, 18],
                 [8, 21, 12, 22]])

# Convert the data array to a pandas DataFrame
df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'Y'])

# Add a constant column for the intercept term
df = sm.add_constant(df)

# Perform stepwise regression
model = sm.OLS(df['Y'], df.drop(columns=['Y']))
result = model.fit()
print(result.summary())

# Export the summary table as a CSV file
summary_table = result.summary().tables[1]
summary_df = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])
summary_df.to_csv('stepwise_regression_summary.csv', index=False)
