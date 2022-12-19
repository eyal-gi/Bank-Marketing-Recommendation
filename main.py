"""
Authors:
Omer Keidar 307887984 & Eyal Ginosar 307830901

Machine Learning and Data Mining Class Final Project
"""
import numpy as np
import os
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, f1_score, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV
from tabulate import tabulate
from random import seed, randint
from dython import nominal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split , KFold
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import VisualizeNN as VisNN
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.metrics import geometric_mean_score
import matplotlib.ticker as maticker



################################
# ---------Read Data-----------
################################

# -------- The read for the understanding and making manipulate on the data------
orig_train_df = pd.read_csv(pjoin(os.getcwd(), 'train.csv'), sep=";")
orig_test_df = pd.read_csv(pjoin(os.getcwd(), 'test.csv'), sep=";")

# -----------After Manipulate and filling missing values----------------------
train_df = pd.read_csv(pjoin(os.getcwd(), 'train_after_manipulate_and_missingVal.csv'))
test_df = pd.read_csv(pjoin(os.getcwd(), 'test_after_manipulate_and_missingVal.csv'))

X_train = train_df.drop(['y'], axis=1)
Y_train = (train_df[['y']])

X_test = test_df.drop(['y'], axis=1)
Y_test = (test_df[['y']])

train_df_xgboost = pd.read_csv(pjoin(os.getcwd(), 'train_after_manipulate_and_missingVal_forXGBOOST.csv'))
test_df_xgboost = pd.read_csv(pjoin(os.getcwd(), 'test_after_manipulate_and_missingVal_forXGBOOST.csv'))

X_train_xgboost = train_df_xgboost.drop(['y'], axis=1)
Y_train_xgboost = (train_df_xgboost[['y']])

X_test_xgboost = test_df_xgboost.drop(['y'], axis=1)
Y_test_xgboost = (test_df_xgboost[['y']])


# -----------With less features----------------------
alt_train_df = pd.read_csv(pjoin(os.getcwd(), 'tree_alt_train.csv'))
alt_test_df = pd.read_csv(pjoin(os.getcwd(), 'tree_alt_test.csv'))

alt_X_train = alt_train_df.drop(['y'], axis=1)
alt_Y_train = (alt_train_df[['y']])

alt_X_test = alt_test_df.drop(['y'], axis=1)
alt_Y_test = (alt_test_df[['y']])
#########################################
# ---------STATIC VARIABLES-----------
#########################################
CAT_VARIABLES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']


# #********************************************************************************************************************#
# ----------------1. Data Understanding---------------------
# #********************************************************************************************************************#

##############################################################
# --Dividing the numerical data into groups and fix the data--
##############################################################

def divide_and_bin(df):  # dividing to groups and make binary instead of yes/no
    """
    After examining the value range of each variable, we chose in a frustrated way and using literature on group
    redistribution.
    In addition, we performed corrections to the data, such as turning the explained variable into 0 and 1
    ( Will not make a deposit - 0 , Will make a deposit - 1)

    :param df: The data to work on it
    :return: Data frame after manipulate
    """

    # --------------------------------------------
    ##### Changing variables to binary #######
    # --------------------------------------------

    y_map = {
        'no': '0',
        'yes': '1'
    }

    df.loc[:, 'y'] = df['y'].map(y_map)  # Change 'y' to binary

    default_map = {
        'no': '0',
        'yes': '1'
    }

    df.loc[:, 'default'] = df['default'].map(default_map)  # Change 'default' to binary

    housing_map = {
        'no': '0',
        'yes': '1'
    }

    df.loc[:, 'housing'] = df['housing'].map(housing_map)  # Change 'housing' to binary

    loan_map = {
        'no': '0',
        'yes': '1'
    }

    df.loc[:, 'loan'] = df['loan'].map(loan_map)  # Change 'loan' to binary

    # --------------------------------------------
    ##### Making groups from variables #######
    # --------------------------------------------

    df['age'] = pd.cut(df['age'], bins=[17, 30, 50, 70, 96], labels=['18-30', '31-50', '51-70', '71-95'])

    df['balance'] = pd.cut(df['balance'],
                           bins=[-8020, 0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                                 110000],
                           labels=['[-8019,0)', '[0,10,000)', '[10,000,20,000)', '[20,000,30,000)', '[30,000,40,000)',
                                   '[40,000,50,000)', '[50,000,60,000)', '[60,000,70,000)', '[70,000,80,000)',
                                   '[80,000,90,000)', '[90,000,100,000)', '[100,000,110,000)'])

    df['day'] = pd.cut(df['day'], bins=[-1, 5, 10, 15, 20, 25, 32], labels=['[0-5)', '[5-10)', '[10-15)', '[15-20)',
                                                                            '[20-25)', '[25-31]'])

    df['duration'] = pd.cut(df['duration'],
                            bins=[-1, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 5000],
                            labels=['[0-300)', '[300-600)', '[600-900)', '[900-1200)', '[1200-1500)',
                                    '[1500-1800)', '[1800-2100)', '[2100-2400)', '[2400-2700)',
                                    '[2700-3000)', '[3000-3300)', '[3300-5000)'])

    # df['campaign'] = pd.cut(df['campaign'], bins=[-1, 10, 20, 30, 40, 50, 60, 70],
    #                         labels=['[1-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)'])

    df['pdays'] = pd.cut(df['pdays'], bins=[-2, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                         labels=['-1', '[0,100)', '[100,200)', '[200,300)', '[300,400)',
                                 '[400,500)', '[500,600)', '[600,700)', '[700,800)',
                                 '[800,900)'])

    # df['previous'] =pd.cut(df['previous'], bins=[-1,25,50,75,100,125,150,175,200,225,250,277] ,labels=['[0-25)','[25-50)',
    #                                                          '[50-75)','[75-100)','[100-125)','[125-150)','[150-175)','[175-200)','[200-225)',
    #                                                          '[225-250)','[250-276)'])
    return df


################################
# ----Apriori probabilities----
################################

def apriori_probs(df):
    """
    Calculates the a priori probabilities for each feature

    :param df: Data frame to work on
    :return: a-priori probabilities for each feature and print them
    """
    age_prob = train_df.groupby('age').size().div(len(train_df))
    job_prob = train_df.groupby('job').size().div(len(train_df))
    marital_prob = train_df.groupby('marital').size().div(len(train_df))
    education_prob = train_df.groupby('education').size().div(len(train_df))
    default_prob = train_df.groupby('default').size().div(len(train_df))
    balance_prob = train_df.groupby('balance').size().div(len(train_df))
    housing_prob = train_df.groupby('housing').size().div(len(train_df))
    loan_prob = train_df.groupby('loan').size().div(len(train_df))
    contact_prob = train_df.groupby('contact').size().div(len(train_df))
    day_prob = train_df.groupby('day').size().div(len(train_df))
    month_prob = train_df.groupby('month').size().div(len(train_df))
    duration_prob = train_df.groupby('duration').size().div(len(train_df))
    campaign_prob = train_df.groupby('campaign').size().div(len(train_df))
    pdays_prob = train_df.groupby('pdays').size().div(len(train_df))
    previous_prob = train_df.groupby('previous').size().div(len(train_df))
    poutcome_prob = train_df.groupby('poutcome').size().div(len(train_df))
    y_prob = train_df.groupby('y').size().div(len(train_df))

    print(age_prob,'\n',job_prob,'\n',marital_prob,'\n',education_prob,'\n',default_prob,'\n',balance_prob,'\n',
          housing_prob,'\n',loan_prob,'\n',contact_prob,'\n',day_prob,'\n',month_prob,'\n',duration_prob,'\n',campaign_prob,
          '\n',pdays_prob,'\n',previous_prob,'\n',poutcome_prob,'\n',y_prob)

    return age_prob,job_prob,marital_prob,education_prob,default_prob,balance_prob,housing_prob,loan_prob,contact_prob,\
           day_prob,month_prob,duration_prob,campaign_prob,pdays_prob,previous_prob,poutcome_prob,y_prob


def corr_heatmap(df, bins):
    if bins is False:
        nominal.associations(df, nominal_columns=CAT_VARIABLES)
    else:
        nominal.associations(df, nominal_columns='all')


# #********************************************************************************************************************#
# ----------------2. Data Preparation---------------------
# #********************************************************************************************************************#

####################################################
# ---------------- Data Cleaning ----------------#
####################################################

def drop_features(df):
    # df = df.drop(['poutcome'], axis=1)  # drop poutcome because there are too many missing values
    df = df.drop(['default'], axis=1)  # drop default because it has low correlation with Y
    df = df.drop(['day'], axis=1)  # drop day because it has low correlation with Y

    # CAT_VARIABLES.remove('poutcome')
    # CAT_VARIABLES.remove('default')
    # CAT_VARIABLES.remove('day')




    # df = df.drop(['marital'], axis=1)  # drop day because it has low correlation with Y
    # df = df.drop(['education'], axis=1)  # drop day because it has low correlation with Y
    df = df.drop(['balance'], axis=1)  # drop day because it has low correlation with Y
    # df = df.drop(['loan'], axis=1)  # drop day because it has low correlation with Y
    df = df.drop(['pdays'], axis=1)  # drop day because it has low correlation with Y
    # df = df.drop(['campaign'], axis=1)  # drop day because it has low correlation with Y
    # CAT_VARIABLES.remove('marital')
    # CAT_VARIABLES.remove('education')
    # CAT_VARIABLES.remove('balance')
    # CAT_VARIABLES.remove('loan')
    # print(CAT_VARIABLES)
    return df


####################################################
# ---------------- Missing Values ----------------#
####################################################
def knn_imputation(df):
    """
    This function gets a data frame, replace all the 'unknown' variables with 'NA', creates dummy variables to all
    categorical features and then impute all missing values using a KNN algorithm.

    :param df: A data frame with 'unknown' values.
    :return:  A data frame where all categorical variables have dummy variables without missing values.
    """
    poutcome = df[['poutcome']]
    df = df.copy().drop(['poutcome'], axis=1)
    df = df.replace(to_replace='unknown', value=np.nan)  # change 'unknown' to NaN
    print(df.isna().sum())  # check which columns has missing values and how many
    df = pd.concat([df, poutcome], axis=1)
    cat_variables = df[CAT_VARIABLES]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=False)  # one-hot-encoding to all categorical variables
    # print(cat_dummies.head())

    # remove old columns and add new columns of one-hot-encoding
    df = df.drop(CAT_VARIABLES, axis=1)
    df = pd.concat([df, cat_dummies], axis=1)
    # print(df.head())
    print(df.info())
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)  # new df without missing values
    return df


########################################################
# ---------------- Update Categorical ----------------#
#########################################################
def update_catVar(cat_var, drop_list):
    for i in range(0, len(drop_list)):
        cat_var.remove(drop_list[i])

    return cat_var



# #********************************************************************************************************************#
# ----------------3. Modeling---------------------
# #********************************************************************************************************************#

####################################################
# ---------------- Threshold Functions ----------------#
####################################################
def pretict_by_threshold(pred_prob , threshold):
    """
    This function get array of probabilities and make a predict by threshold
    :param pred_prob: array of probabilities
    :param threshold: the threshold to predict
    :return: an array of predictions
    """
    pred_p = pd.DataFrame(pred_prob)
    pred_pr = pred_p[[1]]
    pred_pr.where(pred_p > threshold , other = 0 , inplace=True )
    pred_pr.where(pred_p <= threshold , other = 1 , inplace=True)
    pred_pro = pred_pr.loc[:,1]
    pred_val = pred_pro.values

    return pred_val.astype(int)

def find_best_threshold(x_train , y_train , best_model):
    """
    This function get the data and classifier and find the best threshold by calculate G-mean and F1 scores
    :param x_train: Features
    :param y_train: y
    :param best_model: model
    :return: the best threshold for the specific classifier according to G-mean and F1 scores
    """
    y_train=y_train['y']
    trainX , testX , trainY, testY = train_test_split(x_train,y_train, test_size=0.2 , random_state=2, stratify=y_train)
    best_model.fit(trainX,trainY)
    y_prob = best_model.predict_proba(testX)
    y_prob_p = y_prob[:, 1]
    #### G-mean score ####
    fpr , tpr, thresholds_G = roc_curve(testY, y_prob_p)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix_G = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds_G[ix_G], gmeans[ix_G]))
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix_G], tpr[ix_G], marker='o', color='black', label='Best')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    ###### F1 score ######
    precision, recall, thresholds_F = precision_recall_curve(testY,y_prob_p)
    fscore = (2 * precision * recall) / (precision + recall)
    ix_F = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds_F[ix_F], fscore[ix_F]))
    no_skill = len(testY[testY == 1]) / len(testY)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='Logistic')
    pyplot.scatter(recall[ix_F], precision[ix_F], marker='o', color='black', label='Best')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()


####################################################
# ---------------- Modeling general functions ----------------#
####################################################
def holdout_validation(clf, x_train_original, y_train_original, isThreshold=False, threshold=0.5):
    """
    This function validate the model using holdout validation of 80/20. The default function calculate normally using
    the clf.predict() function. Alternatively, it can calculate with an adjusted threshold.
    :param clf: Model classifier
    :param x_train_original: X_train
    :param y_train_original: Y_train
    :param isThreshold: boolean, default is False, if Yes, the validation is calculated by a chosen threshold
    :param threshold: the desired threshold
    :return: scores: A data frame of the scores calculated to measure the model by.
    """

    print("Validate model using holdout:")
    # holdout validation with 80/20
    x_train, x_val, y_train, y_val = train_test_split(x_train_original, y_train_original,
                                                      test_size=0.2, random_state=2, stratify=y_train_original)

    clf.fit(x_train, y_train)

    if isThreshold is False:  # we don't use adjusted threshold
        train_pred = clf.predict(x_train)
        val_pred = clf.predict(x_val)
    else:  # we manually adjust the model;s threshold
        train_pred = pretict_by_threshold(clf.predict_proba(x_train), threshold)
        val_pred = pretict_by_threshold(clf.predict_proba(x_val), threshold)

    # calculate measures
    acc_train = accuracy_score(y_train, train_pred)
    recall_train = recall_score(y_train, train_pred)
    precision_train = precision_score(y_train, train_pred)
    f1_train = f1_score(y_train, train_pred)
    gmean_train = geometric_mean_score(y_train, train_pred)

    acc_val = accuracy_score(y_val, val_pred)
    recall_val = recall_score(y_val, val_pred)
    precision_val = precision_score(y_val, val_pred)
    f1_val = f1_score(y_val, val_pred)
    gmean_val = geometric_mean_score(y_val, val_pred)

    # print results
    print(f"Train Accuracy: {acc_train:.3f}")
    print(f"Train Recall: {recall_train:.3f}")
    print(f"Train Precision: {precision_train:.3f}")
    print(f"Train F1: {f1_train:.3f}")
    print(f"Train g_mean: {gmean_train:.3f}")

    print(f"Validation Accuracy: {acc_val:.3f}")
    print(f"Validation Recall: {recall_val:.3f}")
    print(f"Validation Precision: {precision_val:.3f}")
    print(f"Validation F1: {f1_val:.3f}")
    print(f"Validation g_mean: {gmean_val:.3f}")

    conf_matrix = confusion_matrix(y_true=y_val, y_pred=val_pred)
    print(conf_matrix)

    scores = {
        'acc_train': acc_train, 'recall_train': recall_train, 'precision_train': precision_train, 'f1_train': f1_train,
        'gmean_train': gmean_train,'acc_val': acc_val, 'recall_val': recall_val, 'precision_val': precision_val,
        'f1_val': f1_val, 'gmean_val': gmean_val
    }

    return scores


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----not relevant---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# def cross_validation(clf, x_train_original, y_train_original):
#     train_result = pd.DataFrame()
#     valid_result = pd.DataFrame()
#
#     for train_idx, val_idx in KF.split(x_train_original, y_train_original):
#         print("Fold")
#         x_train = x_train_original.iloc[train_idx]
#         y_train = y_train_original.iloc[train_idx]
#         x_val = x_train_original.iloc[val_idx]
#         y_val = y_train_original.iloc[val_idx]
#
#         clf.fit(x_train, y_train)
#         acc_train = accuracy_score(y_train, clf.predict(x_train))
#         recall_train = recall_score(y_train, clf.predict(x_train))
#         precision_train = precision_score(y_train, clf.predict(x_train))
#         f1_train = f1_score(y_train, clf.predict(x_train))
#
#         acc_val = accuracy_score(y_val, clf.predict(x_val))
#         recall_val = recall_score(y_val, clf.predict(x_val))
#         precision_val = precision_score(y_val, clf.predict(x_val))
#         f1_val = f1_score(y_val, clf.predict(x_val))
#
#         print(f"Train Accuracy: {acc_train:.3f}")
#         print(f"Train Recall: {recall_train:.3f}")
#         print(f"Train Precision: {precision_train:.3f}")
#         print(f"Train F1: {f1_train:.3f}")
#
#         print(f"Validation Accuracy: {acc_val:.3f}")
#         print(f"Validation Recall: {recall_val:.3f}")
#         print(f"Validation Precision: {precision_val:.3f}")
#         print(f"Validation F1: {f1_val:.3f}")
#
#         print(confusion_matrix(y_true=y_val, y_pred=clf.predict(x_val)))
#
#         train_result = train_result.append({'train_acc': acc_train, 'train_recall': recall_train,
#                                             'train_precision': precision_train}, ignore_index=True)
#         valid_result = valid_result.append({'val_acc': acc_val, 'val_recall': recall_val,
#                                             'val_precision': precision_val}, ignore_index=True)
#
#     print(train_result)
#     print(valid_result)
#     avg_acc_train = train_result.mean()
#     avg_acc_val = valid_result.mean()
#     print(avg_acc_train)
#     print(avg_acc_val)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----not relevant---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def print_tuning_results(results):
    """
    This function prints the results of the tuning. The results are printed for each of the measures we test.
    :param results: DataFrame of the tuning results
    """
    pd.set_option('display.max_columns', None)
    results = results.sort_values(by='Val g_mean', ascending=False)
    print("Results by geometric mean:")
    print(results)
    print("Results by f1:")
    print(results.sort_values(by='Val f1', ascending=False))
    print("Results by recall:")
    print(results.sort_values(by='Val recall', ascending=False))
    print("Results by precision:")
    print(results.sort_values(by='Val precision', ascending=False))
    print("Results by acc:")
    print(results.sort_values(by='Val acc', ascending=False))
    rows = len(results)
    print("the number of iterations is: " + str(rows))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-----not relevant---xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# def hyper_params_tuning(grid_search):
#     """
#     This function
#     :param grid_search:
#     :return: void
#     """
#
#     grid_results = pd.DataFrame(grid_search.cv_results_)
#     print("scorer: ", grid_search.scorer_)
#     print("best score: ", grid_search.best_score_)
#     print("best estimator: ", grid_search.best_estimator_)
#     print('The best parameters are:', grid_search.best_params_)
#     print("best estimator: ", grid_search.best_estimator_)
#
#     print("mean_test_f1", grid_results['mean_test_f1'])
#     print("mean_test_acc", grid_results['mean_test_accuracy'])
#     print("mean_test_prec", grid_results['mean_test_precision'])
#     print("mean_test_recall", grid_results['mean_test_recall'])
#
#     y_val = grid_results['mean_test_score']
#     y_train = grid_results['mean_train_score']
#     results_grid_search1 = pd.DataFrame(grid_results).sort_values('rank_test_score')[['params', 'mean_test_score']]
#     results_grid_search2 = pd.DataFrame(grid_results).sort_values('mean_train_score', ascending=False)[
#         ['params', 'mean_train_score']]
#     headers_val = ["Number", "Parameters", "Validation score"]
#     headers_train = ["Number", "Parameters", "Train score"]
#     print(tabulate(results_grid_search1, headers=headers_val, tablefmt="grid"))
#     print(tabulate(results_grid_search2, headers=headers_train, tablefmt="grid"))
#     plt.plot(y_val)
#     plt.ylabel('Validation accuracy', fontsize=10)
#     plt.xlabel('Iteration', fontsize=10)
#     plt.title('Accuracy for each Iteration', fontsize=20)
#     plt.show()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-----not relevant---xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


####################################################
# ---------------- ANN ----------------#
####################################################
def ann_hyperparams_tuning(param_grid, x_train, y_train):
    """
    This function does hyper parameters tuning for the ANN model.
    :param param_grid: The parameters to check
    :param x_train: X train data
    :param y_train: Y train data
    """
    res = []
    params_list = []

    for hidden_size_it in param_grid['hidden_layer_sizes']:
        for activation_it in param_grid['activation']:
            for solver_it in param_grid['solver']:
                for lr_rate_it in param_grid['learning_rate_init']:
                    print("Testing the params: ",str(hidden_size_it), str(activation_it), str(solver_it), str(lr_rate_it))
                    grid_search = MLPClassifier(random_state=42, hidden_layer_sizes=hidden_size_it, activation=activation_it,
                                                learning_rate_init=lr_rate_it, solver=solver_it, max_iter=200)
                    scores = holdout_validation(grid_search, x_train, y_train)
                    params_list.append([str(hidden_size_it), str(activation_it), str(solver_it), str(lr_rate_it)])
                    res.append([str(hidden_size_it), str(activation_it), str(solver_it), str(lr_rate_it),
                                scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],
                                scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],
                                scores['gmean_train'], scores['gmean_val']])

    results = pd.DataFrame(res, columns=['hidden_layers', 'activation', 'solver', 'lr_rate_init',
                                         'Train acc', 'Val acc', 'Train precision', 'Val precision',
                                         'Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print_tuning_results(results)

def mlpc_clf(x_train, y_train, x_test, y_test):
    """
    This function creates a MLPC (ANN) model.
    :param x_train: X_train
    :param y_train: Y_train
    """
    print("Artificial Neural Network Model")
    # Scale numeric variables to 0-1
    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    y_train = y_train['y']

    loss_curve_flag = False  # boolean for plot (YES/NO)
    # ***---basic model:
    ann_model = MLPClassifier(random_state=42)
    # ***---Evaluate basic model with holdout:
    # holdout_validation(ann_model, x_train, y_train)

    # ***---Hyper parameters tuning:
    param_grid = {
        'hidden_layer_sizes': [(57,), (57, 57), (114,), (114, 114), (228, ), (228, 228)],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'activation': ['tanh', 'relu', 'logistic'],
        'learning_rate_init': [0.1, 0.01, 0.001]
    }
    # ann_hyperparams_tuning(param_grid, x_train, y_train)

    # ***---Best model after tuning:
    # ann_model = MLPClassifier(hidden_layer_sizes=(140, 140), activation='tanh', solver='adam', learning_rate_init=0.1, random_state=42)
    ann_model = MLPClassifier(hidden_layer_sizes=(114,), activation='tanh', solver='adam', learning_rate_init=0.1, random_state=42)

    # ***---Find best threshold:
    # find_best_threshold(x_train, alt_Y_train, ann_model)

    # ***---Evaluate best model with holdout using threshold:
    # holdout_validation(ann_model, x_train, y_train, True, threshold=0.096420)
    # holdout_validation(ann_model, x_train, y_train, True, threshold=0.261211)

    # ***---Evaluate on test:
    x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
    y_test = y_test['y']
    # test_model(ann_model, x_train, y_train, x_test, y_test, True, threshold=0.096420)
    test_model(ann_model, x_train, y_train, x_test, y_test, True, threshold=0.261211)

    # visualisations:
    if loss_curve_flag:
        plt.plot(ann_model.loss_curve_)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

        network_structure = np.hstack(([x_train.shape[1]], np.asarray(ann_model.hidden_layer_sizes), 2))
        network = VisNN.DrawNN(network_structure)
        network.draw()

        loss_curve_flag = False



####################################################
# ---------------- Random Forest ----------------#
####################################################

def rf_hyperparams_tuning(param_grid, x_train, y_train):
    res = []
    params_list = []

    for n_est_it in param_grid['n_est']:
        for criterion_it in param_grid['criterion']:
            for max_f_it in param_grid['max_f']:
                    print('the parameters are:', n_est_it, criterion_it, max_f_it)
                    grid_search = RandomForestClassifier(n_estimators=n_est_it, criterion=criterion_it,
                                                max_features=max_f_it , random_state=42)
                    scores = holdout_validation(grid_search, x_train, y_train)
                    params_list.append([str(n_est_it), str(criterion_it), str(max_f_it)])
                    res.append([str(n_est_it), str(criterion_it), str(max_f_it),
                                scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],
                                scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],
                                scores['gmean_train'], scores['gmean_val']])

    results = pd.DataFrame(res, columns=['n_estimators', 'criterion', 'max_features',
                                         'Train acc', 'Val acc', 'Train precision', 'Val precision',
                                         'Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print_tuning_results(results)

def random_forest_clf(x_train, y_train):
    y_train = y_train['y']
    rf_model = RandomForestClassifier(random_state=42)  # basic model
    # holdout_validation(rf_model, x_train, y_train)

    ###_____________** RandomForest Hyper-parameters tuning **______________##
    param_grid = {
        'n_est': [10,20,35,45,52,70,90,110,130,180,200],
        'criterion': ['gini','entropy'],
        'max_f': range(15,45)
    }
    # param_grid = {
    #     'n_est': [10,20,35,45,52,70,90,110,130,180,200],
    #     'criterion': ['gini','entropy'],
    #     'max_f': range(15,45)
    # }

    rf_hyperparams_tuning(param_grid, x_train, y_train)

def rf_best_model(x_train, y_train):
    res = []
    y_train = y_train['y']
    best_rf_model = RandomForestClassifier(n_estimators=35 , criterion='gini', max_features=26,random_state=42)
    holdout_validation(best_rf_model, x_train, y_train)
    threshold_best_rf =0.118571
    scores = holdout_validation(best_rf_model, x_train, y_train , True , threshold_best_rf )
    res.append([scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],scores['gmean_train'], scores['gmean_val']])
    pd.set_option('display.max_columns', None)
    results = pd.DataFrame(res, columns=['Train acc', 'Val acc', 'Train precision', 'Val precision','Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print(results)

####################################################
# --------------------- SVM -----------------------#
####################################################

def svm_hyperparams_tuning(param_grid, x_train, y_train):
    res = []
    params_list = []

    for kernel_it in param_grid['kernel']:
        for c_it in param_grid['c']:
                print('the parameters are:', kernel_it, c_it)
                grid_search = SVC(kernel=kernel_it, C=c_it, random_state=42)
                scores = holdout_validation(grid_search, x_train, y_train)
                params_list.append([str(kernel_it), str(c_it)])
                res.append([str(kernel_it), str(c_it),
                            scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],
                            scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],
                            scores['gmean_train'], scores['gmean_val']])

    results = pd.DataFrame(res, columns=['kernel', 'C','Train acc', 'Val acc', 'Train precision', 'Val precision',
                                         'Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print_tuning_results(results)

def svm_clf(x_train, y_train):
    y_train = y_train['y']
    svm_model = SVC(random_state=42)  # basic model
    holdout_validation(svm_model, x_train, y_train)

    ###_____________** SVM Hyper-parameters tuning **______________##
    param_grid = {
        'kernel': ('linear','poly','rbf','sigmoid'),
        'c': [0.01,0.025,0.1,0.5,1,3,10,100]
    }
    # param_grid = {
    #     'kernel': (['linear']),
    #     'c': [0.01,0.1]
    # }

    # svm_hyperparams_tuning(param_grid, x_train, y_train)

def svm_best_model(x_train, y_train):
    res = []
    y_train = y_train['y']
    best_svm_model = SVC(kernel='rbf', C=100, random_state=42 , probability=True)
    threshold_best_svm =0.100075
    scores = holdout_validation(best_svm_model, x_train, y_train , True , threshold_best_svm )
    res.append([scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],scores['gmean_train'], scores['gmean_val']])
    pd.set_option('display.max_columns', None)
    results = pd.DataFrame(res, columns=['Train acc', 'Val acc', 'Train precision', 'Val precision','Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print(results)


####################################################
# ------------------- XGboost --------------------#
####################################################

def xgboost_hyperparams_tuning(param_grid, x_train, y_train):
    res = []
    params_list = []

    for booster_it in param_grid['booster']:
        for eta_it in param_grid['eta']:
            for max_depth_it in param_grid['max_depth']:
                for min_child_weight_it in param_grid['min_child_weight']:
                    print('the parameters are:', booster_it, eta_it, max_depth_it ,min_child_weight_it)
                    grid_search = XGBClassifier(booster=booster_it, eta=eta_it,max_depth=max_depth_it,min_child_weight=min_child_weight_it,random_state=42)
                    scores = holdout_validation(grid_search, x_train, y_train)
                    params_list.append([str(booster_it), str(eta_it), str(max_depth_it) , str(min_child_weight_it)])
                    res.append([str(booster_it), str(eta_it), str(max_depth_it) , str(min_child_weight_it),
                                scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],
                                scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],
                                scores['gmean_train'], scores['gmean_val']])

    results = pd.DataFrame(res, columns=['booster','eta','max_depth' , 'min_child_weight','Train acc', 'Val acc', 'Train precision', 'Val precision',
                                         'Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print_tuning_results(results)

def xgboost_clf(x_train, y_train):
    y_train = y_train['y']
    xgboost_model = XGBClassifier(random_state=42)  # basic model
    holdout_validation(xgboost_model, x_train, y_train)
    ###_____________** SVM Hyper-parameters tuning **______________##
    param_grid = {
        'booster': ['gbtree','dart'],
        'eta': [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] ,
        'max_depth' : range(4,20,2),
        'min_child_weight' : [1,2,3]

    }


    # xgboost_hyperparams_tuning(param_grid, x_train, y_train)

def xgboost_best_model(x_train, y_train):
    res = []
    y_train = y_train['y']
    best_xgboost_model = XGBClassifier(booster='gbtree', eta=1, max_depth=10 ,min_child_weight=2,random_state=42 )
    holdout_validation(best_xgboost_model, x_train, y_train)
    threshold_best_rf =0.032809
    scores = holdout_validation(best_xgboost_model, x_train, y_train , True , threshold_best_rf )
    res.append([scores['acc_train'], scores['acc_val'], scores['precision_train'], scores['precision_val'],scores['recall_train'], scores['recall_val'], scores['f1_train'], scores['f1_val'],scores['gmean_train'], scores['gmean_val']])
    pd.set_option('display.max_columns', None)
    results = pd.DataFrame(res, columns=['Train acc', 'Val acc', 'Train precision', 'Val precision','Train recall', 'Val recall', 'Train f1', 'Val f1', 'Train g_mean', 'Val g_mean'])
    print(results)


# #********************************************************************************************************************#
# ----------------4. Evaluation---------------------
# #********************************************************************************************************************#

def test_model(clf, x_train, y_train, x_test, y_test, isThreshold=False, threshold=0.5):
    print("Evaluate model on test set")

    clf.fit(x_train, y_train)

    ############### For RF only to make graph of features importance #######
    # feature_importance = clf.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
    # pos = np.arange(sorted_idx.shape[0])
    # plt.barh(y=pos, width=feature_importance[sorted_idx], height=0.5 , align='center')
    # plt.yticks(pos, X_train.columns[sorted_idx], fontsize=6)
    # plt.title('Feature Importance', fontsize = 20)
    # plt.show()
    # ###############################################################################

    if isThreshold is False:  # we don't use adjusted threshold
        train_pred = clf.predict(x_train)
        test_pred = clf.predict(x_test)
    else:  # we manually adjust the model;s threshold
        train_pred = pretict_by_threshold(clf.predict_proba(x_train), threshold)
        test_pred = pretict_by_threshold(clf.predict_proba(x_test), threshold)

    # calculate measures
    acc_train = accuracy_score(y_train, train_pred)
    recall_train = recall_score(y_train, train_pred)
    precision_train = precision_score(y_train, train_pred)
    f1_train = f1_score(y_train, train_pred)
    gmean_train = geometric_mean_score(y_train, train_pred)

    acc_test = accuracy_score(y_test, test_pred)
    recall_test = recall_score(y_test, test_pred)
    precision_test = precision_score(y_test, test_pred)
    f1_test = f1_score(y_test, test_pred)
    gmean_test = geometric_mean_score(y_test, test_pred)

    # print results
    print(f"Train Accuracy: {acc_train:.3f}")
    print(f"Train Recall: {recall_train:.3f}")
    print(f"Train Precision: {precision_train:.3f}")
    print(f"Train F1: {f1_train:.3f}")
    print(f"Train g_mean: {gmean_train:.3f}")

    print(f"Test Accuracy: {acc_test:.3f}")
    print(f"Test Recall: {recall_test:.3f}")
    print(f"Test Precision: {precision_test:.3f}")
    print(f"Test F1: {f1_test:.3f}")
    print(f"Test g_mean: {gmean_test:.3f}")

    print(confusion_matrix(y_true=y_test, y_pred=test_pred))

    scores = {
        'acc_train': acc_train, 'recall_train': recall_train, 'precision_train': precision_train, 'f1_train': f1_train,
        'gmean_train': gmean_train, 'acc_test': acc_test, 'recall_test': recall_test, 'precision_test': precision_test,
        'f1_test': f1_test, 'gmean_test': gmean_test
    }

    return scores

##############################################################################
#----------------------------- IMPROVMENT -----------------------------------
##############################################################################

def vote_clf (x_train , y_train , x_test , y_test):
    column_names = ['y_rf' , 'y_ann' , 'y_xgboost', 'y_svm']
    y_pred_train_df = pd.DataFrame(columns =column_names)
    y_pred_test_df = pd.DataFrame(columns =column_names)

    scaler = MinMaxScaler()
    x_train_for_ann = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_for_ann = pd.DataFrame(scaler.fit_transform(x_test), columns=x_train.columns)
    #### models #####
    rf_model  = RandomForestClassifier(n_estimators=35 , criterion='gini', max_features=26,random_state=42)
    ann_model =  MLPClassifier(hidden_layer_sizes=(140, 140), activation='tanh', solver='adam', learning_rate_init=0.1, random_state=42)
    xgboost_model = XGBClassifier(booster='gbtree', eta=1, max_depth=10 ,min_child_weight=2,random_state=42 )
    svm_model = SVC(kernel='rbf', C=100, random_state=42, probability=True)
    ###################
    ###### thresholds #######
    threshold_best_rf = 0.118571
    threshold_best_ann=0.096420
    threshold_best_xgboost = 0.032809
    threshold_best_svm = 0.100075
    ###########################

    ########### fitting ############
    rf_model.fit(x_train,y_train)
    ann_model.fit(x_train_for_ann,y_train)
    xgboost_model.fit(X_train_xgboost,Y_train_xgboost)
    svm_model.fit(x_train,y_train)
    ##################################

    train_pred_rf = pretict_by_threshold(rf_model.predict_proba(x_train), threshold_best_rf)
    test_pred_rf = pretict_by_threshold(rf_model.predict_proba(x_test), threshold_best_rf)
    y_pred_train_df['y_rf'] = train_pred_rf
    y_pred_test_df['y_rf'] = test_pred_rf

    train_pred_ann = pretict_by_threshold(ann_model.predict_proba(x_train_for_ann), threshold_best_ann)
    test_pred_ann = pretict_by_threshold(ann_model.predict_proba(x_test_for_ann), threshold_best_ann)
    y_pred_train_df['y_ann'] = train_pred_ann
    y_pred_test_df['y_ann'] = test_pred_ann


    train_pred_xgboost = pretict_by_threshold(xgboost_model.predict_proba(X_train_xgboost), threshold_best_xgboost)
    test_pred_xgboost = pretict_by_threshold(xgboost_model.predict_proba(X_test_xgboost), threshold_best_xgboost)
    y_pred_train_df['y_xgboost'] = train_pred_xgboost
    y_pred_test_df['y_xgboost'] = test_pred_xgboost

    train_pred_svm = pretict_by_threshold(svm_model.predict_proba(x_train), threshold_best_svm)
    test_pred_svm = pretict_by_threshold(svm_model.predict_proba(x_test), threshold_best_svm)
    y_pred_train_df['y_svm'] = train_pred_svm
    y_pred_test_df['y_svm'] = test_pred_svm


    y_pred_train_df["sum"] = y_pred_train_df.sum(axis=1)
    y_pred_train_df['y'] = np.where(y_pred_train_df['sum'] > 2, 1, 0)

    y_pred_test_df["sum"] = y_pred_test_df.sum(axis=1)
    y_pred_test_df['y'] = np.where(y_pred_test_df['sum'] > 2, 1, 0)

    y_predict_train = pd.DataFrame(y_pred_train_df['y'])
    y_predictions_train = y_predict_train.iloc[:,0]
    y_predictions_train = y_predictions_train.values
    y_predictions_train= y_predictions_train.astype(int)

    y_predict_test = pd.DataFrame(y_pred_test_df['y'])
    y_predictions_test = y_predict_test.iloc[:, 0]
    y_predictions_test = y_predictions_test.values
    y_predictions_test = y_predictions_test.astype(int)

    # # calculate measures
    acc_train = accuracy_score(y_train, y_predictions_train)
    recall_train = recall_score(y_train, y_predictions_train)
    precision_train = precision_score(y_train, y_predictions_train)
    f1_train = f1_score(y_train, y_predictions_train)
    gmean_train = geometric_mean_score(y_train, y_predictions_train)
    #
    acc_test = accuracy_score(y_test, y_predictions_test)
    recall_test = recall_score(y_test, y_predictions_test)
    precision_test = precision_score(y_test, y_predictions_test)
    f1_test = f1_score(y_test, y_predictions_test)
    gmean_test = geometric_mean_score(y_test, y_predictions_test)
    #
    # # print results
    print(f"Train Accuracy: {acc_train:.3f}")
    print(f"Train Recall: {recall_train:.3f}")
    print(f"Train Precision: {precision_train:.3f}")
    print(f"Train F1: {f1_train:.3f}")
    print(f"Train g_mean: {gmean_train:.3f}")
    #
    print(f"Test Accuracy: {acc_test:.3f}")
    print(f"Test Recall: {recall_test:.3f}")
    print(f"Test Precision: {precision_test:.3f}")
    print(f"Test F1: {f1_test:.3f}")
    print(f"Test g_mean: {gmean_test:.3f}")
    #
    print(confusion_matrix(y_true=y_test, y_pred=y_predictions_test))
    #
    scores = {
        'acc_train': acc_train, 'recall_train': recall_train, 'precision_train': precision_train, 'f1_train': f1_train,
        'gmean_train': gmean_train, 'acc_test': acc_test, 'recall_test': recall_test, 'precision_test': precision_test,
        'f1_test': f1_test, 'gmean_test': gmean_test
    }

    return scores


def visual_for_undr():
    # plt.hist(orig_train_df['duration'], bins=200, density=True)
    # plt.title("Duration histogram", fontsize=20)
    # plt.xlabel('Duration', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # sns.distplot(orig_train_df['duration'], hist=False, kde=True)
    # plt.show()

    # ax = orig_train_df['education'].value_counts().plot(kind='bar' , title='Education Plot',figsize=(15, 10),  fontsize=12, color=['darkblue' , 'darkorange' , 'darkred' , 'darkgreen'])
    # plt.xticks(rotation = 360 , horizontalalignment="center")
    # ax.set_xlabel("Education", fontsize=12)
    # ax.set_ylabel("Amount", fontsize=12)
    # plt.show()
    background_color = "#fbfbfb"

    fig = plt.figure(figsize=(24, 16), dpi=100)
    fig.patch.set_facecolor(background_color)
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.35, hspace=0.4)
    plt0 = fig.add_subplot(gs[0, 0])
    plt1 = fig.add_subplot(gs[0, 1])
    plt2 = fig.add_subplot(gs[0, 2])
    plt3 = fig.add_subplot(gs[1, 0])
    plt4 = fig.add_subplot(gs[1, 1])
    plt5 = fig.add_subplot(gs[1, 2])
    plt6 = fig.add_subplot(gs[2, 0])
    plt7 = fig.add_subplot(gs[2, 1])
    plt8 = fig.add_subplot(gs[2, 2])

    ####### Age ########
    plt0.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    train = pd.DataFrame(orig_train_df['age'])
    sns.kdeplot(train['age'], ax=plt0, color="darkblue", shade=True, label="Train")
    plt0.set_title('Age', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt0.set_ylabel('Density')
    plt0.set_xlabel('Age' )

    ####### Month ########
    train = pd.DataFrame(orig_train_df["month"].value_counts())
    train["Percentage"] = train["month"].apply(lambda x: x / sum(train["month"]) * 100)
    train = train.sort_index()
    plt1.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    plt1.plot(train.index, train["Percentage"], zorder=3, color="darkblue", marker='o')
    plt1.yaxis.set_major_formatter(maticker.PercentFormatter())
    plt1.set_title('Month', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt1.set_ylabel('Density')
    plt1.set_xlabel('Month')

    ###### Loan ######
    train = pd.DataFrame(orig_train_df["loan"].value_counts())
    train["Percentage"] = train["loan"].apply(lambda x: x / sum(train["loan"]) * 100)
    plt2.barh(train.index, train['Percentage'], color="darkblue", zorder=3, height=0.6)
    plt2.xaxis.set_major_formatter(maticker.PercentFormatter())
    plt2.set_title('Loan', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt2.set_ylabel('Have Loan')
    plt2.set_xlabel('Density')

    ###### Poutcome ######
    train = pd.DataFrame(orig_train_df["poutcome"].value_counts())
    train["Percentage"] = train["poutcome"].apply(lambda x: x / sum(train["poutcome"]) * 100)
    plt3.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    plt3.bar(train.index, height=train["Percentage"], zorder=3, color="darkblue", width=0.4)
    plt3.yaxis.set_major_formatter(maticker.PercentFormatter())
    plt3.set_title('Poutcome', fontsize=14, fontweight='bold', fontfamily='serif')
    plt3.set_ylabel('Density')
    plt3.set_xlabel('Poutcome')

    ###### Duration ######
    plt4.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    train = pd.DataFrame(orig_train_df['duration'])
    sns.kdeplot(train['duration'], ax=plt4, color='darkblue', shade=True, label="Train")
    plt4.set_title('Duration', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt4.set_ylabel('Density')
    plt4.set_xlabel('Duration')

    ###### Education ######
    train = pd.DataFrame(orig_train_df["education"].value_counts())
    train["Percentage"] = train["education"].apply(lambda x: x / sum(train["education"]) * 100)
    train = train.sort_index()
    plt5.bar(train.index, height=train["Percentage"], zorder=3, color="darkblue", width=0.04)
    plt5.scatter(np.arange(len(train.index)), train["Percentage"], zorder=3, s=100, color="blue")
    plt5.yaxis.set_major_formatter(maticker.PercentFormatter())
    plt5.set_title('Education', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt5.set_ylabel('Density')
    plt5.set_xlabel('Education')

    ###### Job ######
    train = pd.DataFrame(orig_train_df["job"].value_counts())
    train["Percentage"] = train["job"].apply(lambda x: x / sum(train["job"]) * 100)
    plt6.barh(train.index, train["Percentage"], zorder=3, color="darkblue", height=0.4)
    plt6.xaxis.set_major_formatter(maticker.PercentFormatter())
    plt6.set_title('Job', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt6.set_ylabel('Density')
    plt6.set_xlabel('Job')

    ###### Pdays ######
    plt7.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    train = pd.DataFrame(orig_train_df["pdays"])
    sns.kdeplot(train['pdays'], ax=plt7, color="darkblue", shade=True, label="Train")
    plt7.set_title('Pdays', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt7.set_ylabel('Density')
    plt7.set_xlabel('Pdays')


    ###### Campaign ######
    plt8.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    train = pd.DataFrame(orig_train_df['campaign'])
    sns.kdeplot(train['campaign'], ax=plt8, color="darkblue", shade=True, label="Train")
    plt8.set_title('Campaign', fontsize = 14 ,fontweight='bold',fontfamily='serif')
    plt8.set_ylabel('Density')
    plt8.set_xlabel('Campaign')

    for i in range(0, 9):
        locals()["plt" + str(i)].set_facecolor(background_color)

    for i in range(0, 9):
        locals()["plt" + str(i)].tick_params(axis=u'both', which=u'both', length=0)

    for s in ["top", "right", "left"]:
        for i in range(0, 9):
            locals()["plt" + str(i)].spines[s].set_visible(False)

    plt.show()

###########################################################################################
#------------------------------------main------------------------------------------------
###########################################################################################

if __name__ == '__main__':

    apriori_probs(orig_test_df)
    test_df.info() # Check for missing values
    # corr_heatmap(train_df, bins=False)  # heatmap before binning
    #######prepare data########
    divide_and_bin(orig_train_df)
    divide_and_bin(orig_test_df)
    CAT_VARIABLES = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                      'month', 'duration', 'pdays', 'poutcome']
    orig_train_df = drop_features(orig_train_df)
    orig_test_df = drop_features(orig_test_df)
    #
    # CAT_VARIABLES = ['age', 'job',  'marital', 'education', 'housing', 'loan', 'contact',
    #                  'month', 'duration', 'poutcome']
    # orig_train_df = knn_imputation(orig_train_df)
    # orig_test_df = knn_imputation(orig_test_df)
    #
    # orig_train_df.to_csv("tree_alt_train.csv")
    # orig_test_df.to_csv("tree_alt_test.csv")


    # train_df_for_dum = train_df.copy().drop(['y'],axis=1)
    # train_df_for_dum = train_df_for_dum.drop(['campaign'], axis=1)
    # train_df_for_dum = train_df_for_dum.drop(['previous'], axis=1)
    # train_df_for_dum.to_csv('train_data_final.csv')
    ###############end prepare#####################
    # Update the categorical variables after binning
    # CAT_VARIABLES = update_catVar(CAT_VARIABLES, ['campaign', 'previous'])

    # corr_heatmap(train_df.copy().astype(object), bins=True)   # heatmap after binning

    # print(CAT_VARIABLES)

    # mlpc_clf(X_train, Y_train, X_test, Y_test)

    # xgboost_clf(X_train,Y_train)
    #  random_forest_clf(X_train, Y_train)
    # rf_best_model(X_train, Y_train)
    # xgboost_best_model(X_train, Y_train)
    # svm_best_model(X_train, Y_train)
    ############# find threshold ########################
    ###### RF ######
    # best_rf_model = RandomForestClassifier(n_estimators=35, criterion='gini', max_features=26, random_state=42)
    # find_best_threshold(X_train, Y_train , best_rf_model)
    # threshold_best_rf = 0.118571
    # test_model(best_rf_model, X_train, Y_train['y'], X_test, Y_test['y'], True, threshold_best_rf)
    ###### SVM ######
    # best_svm_model = SVC(kernel='rbf', C=100, random_state=42 , probability=True)
    # find_best_threshold(X_train, Y_train , best_svm_model)
    # threshold_best_svm = 0.100075
    # test_model(best_svm_model, X_train, Y_train['y'], X_test, Y_test['y'], True, threshold_best_svm)
    ###### XGboost ######
    # best_xgboost_model = XGBClassifier(booster='gbtree', eta=1, max_depth=10 ,min_child_weight=2,random_state=42 )
    # find_best_threshold(X_train, Y_train , best_xgboost_model)
    # threshold_best_xgboost = 0.032809
    # test_model(best_xgboost_model, X_train, Y_train['y'], X_test, Y_test['y'], True, threshold_best_xgboost)

    ######################end model with threshold#################

    # vote_clf (X_train, Y_train['y'], X_test, Y_test['y'])
    # visual_for_undr()