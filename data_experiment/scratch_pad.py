import pandas as pd

df = pd.DataFrame([10, 20, 15, 30, 45])

print(df)
print(df.shift(-2))

for i in range(0,4):
    print(i)
