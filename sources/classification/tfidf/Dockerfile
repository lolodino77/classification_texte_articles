FROM python:3.8-slim
WORKDIR /classification_texte_articles_version_objet

COPY sources/classification/tfidf/requirements.txt sources/classification/tfidf/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r ./sources/classification/tfidf/requirements.txt

COPY data/input/merged_corpus/corpus_sceptic_theist.csv data/input/merged_corpus/corpus_sceptic_theist.csv
COPY data/input/merged_corpus/labelEncoder_category_sceptic_theist.joblib data/input/merged_corpus/labelEncoder_category_sceptic_theist.joblib
COPY sources/lib_general.py sources/lib_general.py
COPY sources/main_test.py sources/main_test.py
COPY sources/classification/lib_classification.py sources/classification/lib_classification.py
COPY sources/classification/tfidf/3_train_test_best_model.py sources/classification/tfidf/3_train_test_best_model.py
RUN mkdir -p data/output/sceptic_theist/

# RUN python 3_train_test_best_model.py corpus_sceptic_theist.csv
WORKDIR ./sources/classification/tfidf
CMD ["python", "-u", "3_train_test_best_model.py", "corpus_sceptic_theist.csv"]

# CMD ["python", "-u", "sources/classification/tfidf/3_train_test_best_model.py", "corpus_sceptic_theist.csv"]

#  > [10/11] RUN python 3_train_test_best_model.py corpus_sceptic_theist.csv:
# #14 8.600 sys.path = ['/sources/classification/tfidf', '/usr/local/lib/python38.zip', '/usr/local/lib/python3.8', '/usr/local/lib/python3.8/lib-dynload', '/usr/local/lib/python3.8/site-packages']
# #14 8.601 Traceback (most recent call last):
# #14 8.601   File "3_train_test_best_model.py", line 18, in <module>
# #14 8.601     from lib_classification import *
# #14 8.601 ModuleNotFoundError: No module named 'lib_classification'
# ------
# executor failed running [/bin/sh -c python 3_train_test_best_model.py corpus_sceptic_theist.csv]: exit code: 1

# commandes docker a entrer dans le terminal :
# cd C:\Users\eupho\OneDrive\Documents\perso\projets\classification_texte_articles_version_objet
# docker build -t img_test -f sources/classification/tfidf/Dockerfile .
# docker run -v /data/output/sceptic_theist:/data/output/sceptic_theist img_test

# pour supprimer :
# docker rm -f conteneur
# docker image rm img_test