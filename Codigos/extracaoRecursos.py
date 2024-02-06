from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Separação dos conjuntos em treino e teste
df_liar = pd.read_csv('./dataframes/dataframe_liar.csv')
df_covid = pd.read_csv('./dataframes/dataframe_covid.csv')
df_gm = pd.read_csv('./dataframes/dataframe_gm.csv')

df_liar_train, df_liar_test, df_liar_train_class, df_liar_test_class = train_test_split(df_liar['Token'], df_liar['Label'], test_size=0.25, random_state=42)
df_covid_train, df_covid_test, df_covid_train_class, df_covid_test_class = train_test_split(df_covid['Token'], df_covid['Label'], test_size=0.25, random_state=42)
df_gm_train, df_gm_test, df_gm_train_class, df_gm_test_class = train_test_split(df_gm['Token'], df_gm['Label'], test_size=0.25, random_state=42)


#Bag-of-Words (TF)
TF = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')

#Liar
TF_df_liar_train = TF.fit_transform(df_liar_train)
TF_df_liar_test = TF.transform(df_liar_test)

sparse.save_npz("TF_df_liar_train.npz", TF_df_liar_train)
sparse.save_npz("TF_df_liar_test.npz", TF_df_liar_test)

#Covid
TF_df_covid_train = TF.fit_transform(df_covid_train)
TF_df_covid_test = TF.transform(df_covid_test)

sparse.save_npz("TF_df_covid_train.npz", TF_df_covid_train)
sparse.save_npz("TF_df_covid_test.npz", TF_df_covid_test)

#GM
TF_df_gm_train = TF.fit_transform(df_gm_train)
TF_df_gm_test = TF.transform(df_gm_test)

sparse.save_npz("TF_df_gm_train.npz", TF_df_gm_train)
sparse.save_npz("TF_df_gm_test.npz", TF_df_gm_test)


#TF-IDF
TFIDF = TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, stop_words='english')

#Liar
TFIDF_df_liar_train = TFIDF.fit_transform(df_liar_train)
TFIDF_df_liar_test = TFIDF.transform(df_liar_test)

sparse.save_npz("TFIDF_df_liar_train.npz", TFIDF_df_liar_train)
sparse.save_npz("TFIDF_df_liar_test.npz", TFIDF_df_liar_test)

#Covid
TFIDF_df_covid_train = TFIDF.fit_transform(df_covid_train)
TFIDF_df_covid_test = TFIDF.transform(df_covid_test)

sparse.save_npz("TFIDF_df_covid_train.npz", TFIDF_df_covid_train)
sparse.save_npz("TFIDF_df_covid_test.npz", TFIDF_df_covid_test)

#GM
TFIDF_df_gm_train = TFIDF.fit_transform(df_gm_train)
TFIDF_df_gm_test = TFIDF.transform(df_gm_test)

sparse.save_npz("TFIDF_df_gm_train.npz", TFIDF_df_gm_train)
sparse.save_npz("TFIDF_df_gm_test.npz", TFIDF_df_gm_test)
