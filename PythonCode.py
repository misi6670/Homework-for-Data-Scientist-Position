import pandas as pd

features = pd.read_csv (r'features.csv')
sales = pd.read_csv (r'sales.csv')
stores = pd.read_csv (r'stores.csv')

data = pd.merge(features, sales, on=["Store", "Date", "IsHoliday"])
data = pd.merge(data, stores, on="Store")
data = data.astype({"Store": int, "Temperature": float, "Fuel_Price": float,
                    "MarkDown1": float, "MarkDown2": float, "MarkDown3": float, 
                    "MarkDown4": float, "MarkDown5": float, "CPI": float, "Unemployment": float,
                    "IsHoliday": bool, "Dept": int, "Weekly_Sales": float, "Type": str,
                    "Size": int})
data["Date"] = pd.to_datetime(data["Date"], format = "%d/%m/%Y")
#data = data.fillna(0)
data['IsProfit'] = (data['Weekly_Sales'] >= 0)

import datetime as dt
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
print(data)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 8))
datacor = data[["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", 
                "MarkDown5", "CPI", "Unemployment", "Weekly_Sales", "Size"]]
heatmap = sns.heatmap(datacor.corr()[["Weekly_Sales"]].sort_values(
    by="Weekly_Sales",ascending=False), vmin=-1, vmax=1, annot=True, fmt = '.4f', cmap='BrBG')
heatmap.set_title('Features Correlating with Weekly Sales', fontdict={'fontsize':18}, pad=16);

profit = data.loc[(data['IsProfit'] == True)]
loss = data.loc[(data['IsProfit'] == False)]

plt.figure(figsize=(6, 8))
datacor = profit[["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", 
                  "MarkDown5", "CPI", "Unemployment", "Weekly_Sales", "Size"]]
heatmap = sns.heatmap(datacor.corr()[["Weekly_Sales"]].sort_values(
    by="Weekly_Sales", ascending=False), vmin=-1, vmax=1, annot=True, fmt = '.4f', cmap='BrBG')
heatmap.set_title('Features Correlating with Profitable Weekly Sales', 
                  fontdict={'fontsize':18}, pad=16);

plt.figure(figsize=(6, 8))
datacor = loss[["Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", 
                "MarkDown5", "CPI", "Unemployment", "Weekly_Sales", "Size"]]
heatmap = sns.heatmap(datacor.corr()[["Weekly_Sales"]].sort_values(
    by="Weekly_Sales", ascending=False), vmin=-1, vmax=1, annot=True, fmt = '.3f', cmap='BrBG')
heatmap.set_title('Features Correlating with Unprofitable Weekly Sales', 
                  fontdict={'fontsize':18}, pad=16);

def check(data, group, columns):
  print(data[columns].groupby(group).agg(['mean', 'std']).round(2))

check(data, ['Type', "Year"], ["Year", 'Weekly_Sales', "Type", "Size"])

check(data.dropna(), ["IsProfit",'Type', "Year", "IsHoliday"], 
      ["IsProfit","Year", 'Weekly_Sales', "Type", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4",
       "MarkDown5", "IsHoliday"])


check(data, ["IsProfit",'Type', "Year"], ["IsProfit","Year", 'Weekly_Sales', "Type", 
                                                   "CPI", "Fuel_Price", "Unemployment"])

check(data, ["IsProfit","Month","IsHoliday"], ["IsProfit","IsHoliday","Month",'Weekly_Sales', 
                                               "Temperature"])
