import joblib
import timeit
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.sparse import load_npz

grid_RF = { 
    'bootstrap': [True, False],
    'max_depth': [5, 10, 30],
    'max_features': ['auto', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 800],
    'criterion' :['gini', 'entropy']
}

gridcv_RF = GridSearchCV(estimator = RandomForestClassifier(), param_grid = grid_RF, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)

#(dataframe, String, String)
def rf_funcao(df,fe_name,df_name):
    df_train, df_test, df_train_class, df_test_class = train_test_split(df['Token'], df['Label'], test_size=0.25, random_state=42)

    FE_df_train = load_npz('../npz/' + fe_name + '_' + df_name + '_train.npz')
    FE_df_test = load_npz('../npz/' + fe_name + '_' + df_name + '_test.npz')

    start_train = timeit.default_timer()
    gridcv_RF.fit(FE_df_train, df_train_class)
    stop_train = timeit.default_timer()

    start_predict = timeit.default_timer()
    FE_df_pred = gridcv_RF.predict(FE_df_test)
    stop_predict = timeit.default_timer()

    cv_results = gridcv_RF.cv_results_['split0_test_score'],
    gridcv_RF.cv_results_['split1_test_score'],
    gridcv_RF.cv_results_['split2_test_score'],
    gridcv_RF.cv_results_['split3_test_score'],
    gridcv_RF.cv_results_['split4_test_score']
    mean_score = np.mean(cv_results)
    std_score = np.std(cv_results)

    print("Mean cross-validation score: {:.2f}".format(mean_score))
    print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

    print("Best hyperparameters: {}".format(gridcv_RF.best_params_))
    print("Best score: {}".format(gridcv_RF.best_score_))

    print(classification_report(df_test_class, FE_df_pred))
    print("Accuracy: ", accuracy_score(df_test_class, FE_df_pred))
    print("f1_score: ", f1_score(df_test_class, FE_df_pred, average='micro'))
    print("precision_score: ", precision_score(df_test_class, FE_df_pred, average='micro'))
    print("recall_score: ", recall_score(df_test_class, FE_df_pred, average='micro'))

    print('Train Time: ', stop_train - start_train)
    print('Prediction Time : ', stop_predict - start_predict)
    print('Inference Time: ', (stop_predict - start_predict)/FE_df_test.shape[0])

    cm = confusion_matrix(df_test_class, FE_df_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('../matrizConfusao/' + fe_name + '_' + df_name + '_RF_matriz.png',format='png')
    plt.figure()

    np.savetxt(fname="../predicao/" + fe_name + "_" + df_name + "_RF_pred.csv", X=FE_df_pred, fmt='%s' , delimiter=",")
    joblib.dump(gridcv_RF, '../gridcv/' + fe_name + '_' + df_name + '_RF.sav')


print("\n \n \n Iniciando aplicação \n \n")

df_gm = pd.read_csv('../dataframes/dataframe_gm.csv')
df_covid = pd.read_csv('../dataframes/dataframe_covid.csv')
df_liar = pd.read_csv('../dataframes/dataframe_liar.csv')
print("\n Carregou dataframes \n")

###################

print("\n Começando a rodar conjunto de dados GM")

print("\n Começando a rodar TF \n")
rf_funcao(df_gm,"TF","df_gm",False)

print("\n Terminou TF \n")

print("\n Começando a rodar TFIDF \n")
rf_funcao(df_gm,"TFIDF","df_gm",False)

print("\n Terminou TFIDF \n")

print("\n \n FIM DO GM \n \n \n")

###################

print("\n Começando a rodar conjunto de dados COVID")

print("\n Começando a rodar TF \n")
rf_funcao(df_covid,"TF","df_covid",False)

print("\n Terminou TF \n")

print("\n Começando a rodar TFIDF \n")
rf_funcao(df_covid,"TFIDF","df_covid",False)

print("\n Terminou TFIDF \n")

print("\n \n FIM DO COVID \n \n \n")

###################

print("\n Começando a rodar conjunto de dados LIAR")

print("\n Começando a rodar TF \n")
rf_funcao(df_liar,"TF","df_liar",False)

print("\n Terminou TF \n")

print("\n Começando a rodar TFIDF \n")
rf_funcao(df_liar,"TFIDF","df_liar",False)

print("\n Terminou TFIDF \n")

print("\n \n FIM DO LIAR \n \n \n")
