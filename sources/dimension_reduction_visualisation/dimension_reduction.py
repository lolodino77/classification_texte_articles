import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from pathlib import Path, PureWindowsPath
from inspect import getsourcefile
from os.path import abspath

pd.set_option('display.max_colwidth', 60)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = PureWindowsPath(os.path.dirname(os.getcwd()) + "\\data\\data.parquet")
path = path.as_posix()
corpus = pd.read_parquet(path) #engine="fastparquet"
print(corpus)

#Verifier qu'il n'y a pas d'id en doublon
print(corpus.id.duplicated().any())
print(corpus.index.duplicated().any())

sb.histplot(data=corpus, x="length")
plt.xlim(0, 600)
plt.show()

X = corpus["message_preprocessed"]
y = corpus["category"]
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

### Reduction de dimensions pour potentiellement observer des clusters
print("number of dimensions at the beginninig =", X_tfidf.shape)

# TruncatedSVD
# Pas de PCA car ne prend pas de matrices creuses en entrees
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)

# TSNE
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, verbose=1)
tsne_results = tsne.fit_transform(X_tfidf)
X_embedded.shape
svd.fit(X_tfidf)



import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
df = pd.read_parquet("data.parquet", engine="fastparquet")
sb.histplot(data=df.iloc[0:100], x="length")
plt.show(block=True)
