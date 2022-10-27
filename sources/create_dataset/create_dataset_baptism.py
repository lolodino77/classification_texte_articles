import pandas as pd
pd.set_option('display.max_colwidth', 100)

res = open("corpus_baptism.txt", "r").read().split("\n\n")
res=[elt for elt in res if len(elt) > 1]

message = res
length = [len(elt) for elt in res]
list_of_rows = list(zip(message, length))

df = pd.DataFrame(list_of_rows, columns=["message", "length"])
print(df.head(20))
print(df.shape)

filename = "dataset_baptism.csv"
df.to_csv(filename, index=False)
