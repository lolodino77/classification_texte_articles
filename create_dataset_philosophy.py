import pandas as pd

res = open("corpus_philosophy.txt", "r").read().split("\n")
res=[elt for elt in res if len(elt) > 1]

message = res
length = all_len = [len(elt) for elt in res]
list_of_rows = list(zip(message, length))

df = pd.DataFrame(list_of_rows, columns=["message", "length"])
print(df.head(20))
print(df.shape)

filename = "dataset_philosophy.csv"
df.to_csv(filename, index=False)