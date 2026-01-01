import pandas as pd

# create a sample DataFrame
From = ['P1', 'P2', 'P3', 'P4']
To = ['D1', 'D2', 'D3', 'D4']
data = pd.DataFrame({
    "D1": [0, 0, 40, 0],
    "D2": [0, 0, 0, 0],
    "D3": [40, 10, 0, 0],
    "D4": [0, 10, 0, 0]
}, index=From)
data.to_csv('transportation_demand_M4.csv')
print(data.head(5))
