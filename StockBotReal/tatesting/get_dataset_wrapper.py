import get_dataset

f = get_dataset.get_historical_data("SPY")
print(f)
f.to_csv("./datasets/{}.csv".format("SPY"), index=True)

