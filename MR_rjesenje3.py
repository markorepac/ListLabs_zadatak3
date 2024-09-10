import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_validate ,ParameterGrid, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import joblib
from matplotlib.colors import ListedColormap

PLOT = 1 # Change to 1 to output plots or 0 to not
BIG = (14,8)
SMALL = (7,4)
         
def print_results(model, pl=PLOT, sc=1, model_name="this", suffix=""):
    '''
    Function which is used to print and return classification report,
    confusion matrix and score. sc is argument if scaled data should be used.
    '''
    if sc:
        y_pred = model.predict(X_test_scaled)
        final_score = model.score(X_test_scaled,y_test)
    else:
        y_pred = model.predict(X_test)
        final_score = model.score(X_test,y_test)
    confmat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=5, output_dict=True)
    report_txt = classification_report(y_test, y_pred, digits=5)
    print(f'\nAccuracy score for {model_name}{suffix} model on unseen test data is: {final_score}')
    print(f'\n\n Confusion matrix for {model_name}{suffix} model below has only {len(y_test) - np.diag(confmat).sum()} misses\n\n {confmat} \n')
    print(f'\n\n Classification report for {model_name}{suffix} model: \n {report_txt}')
    return (report, confmat, final_score)



df = pd.read_csv('./training_data.csv')

# Extracting features and targets
y = df['Class'].values
X = df.drop('Class',axis=1).values


# Spliting data in Train, Validation, Test (60% 20% 20%) sets by two times applying train_test_split
X_tv, X_test, y_tv, y_test = train_test_split(X,y, test_size=0.2, random_state=5, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv,y_tv, test_size=0.25, random_state=5, stratify = y_tv)


# PART 1. Training and validation with for loop and ParameterGrid for KNN
print("Starting part one, Training and validation with for loop and ParameterGrid for KNN\n")

scaler = StandardScaler()

# Creating scaled versions of features for cases when needed
X_train_scaled = scaler.fit_transform(X_train)
X_tv_scaled = scaler.transform(X_tv)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Choosin K-Nearest neighbors for a base model with default values
knn = KNeighborsClassifier()
knn.fit(X_tv,y_tv)


# Checking a score for default values. Returns accuracy of 99.80% which is already very good
print(f'Accuracy for base case of K nearest neighbors is {knn.score(X_test,y_test):.5f}')


# Checking if scaling is further improving the score. 
# Further improve to 99.97%. We continue with scaled values
knn.fit(X_tv_scaled, y_tv)
print(f'\nAccuracy for scaled base case of K nearest neighbors is {knn.score(X_test_scaled,y_test):.5f}')


# Fine tuning of hyperparameters for K nearest neighbors with train-validation-test sets
# Selected parameters are n_neighbors and weights
# using only grid and for loop for tuning
knn_grid = {'n_neighbors': range(1,21), 'weights':['uniform','distance']}
knn_all_params = []
knn_best_params = {}
knn_best = 0

print("\nCalculating KNN wiht grid parameters.....")
for params in ParameterGrid(knn_grid):
    p = params.copy()
    knn.set_params(**params)
    knn.fit(X_train_scaled,y_train)
    # During hyperparameter tuning we use validation set
    score = knn.score(X_val_scaled,y_val)
    p['score'] = score
    knn_all_params.append(p)
    if score > knn_best:
        knn_best = score
        knn_best_params = params
        
print(f'\nAfter tuning best parameters found for KNN is {knn_best_params} with accuracy of {knn_best}')

# Converting scores into a dataframe and spliting for two different weights    
knn_score_df = pd.DataFrame(knn_all_params)
knn_uni_df = knn_score_df[knn_score_df['weights']=='uniform']
knn_dist_df = knn_score_df[knn_score_df['weights']=='distance']

# Ploting the results of hyperparameter tuning. Set true for ploting and saving
if PLOT:
    plt.figure(1, figsize=SMALL)
    plt.set_cmap('copper')
    plt.plot(knn_uni_df['n_neighbors'].values,knn_uni_df['score'].values)
    plt.plot(knn_dist_df['n_neighbors'].values,knn_dist_df['score'].values)
    plt.title("Hypertuning results for KNN")
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy score")
    plt.legend(['uniform', 'distance'])
    plt.savefig('Figure1_knn_tuning results.png')
    plt.show()
    
# Retraining KNN model with the best parameters and final testing with unseen test dataset
knn.set_params(**knn_best_params)
knn.fit(X_tv_scaled,y_tv)


# Printing final results for KNN and ploting if second arg is True
print_results(knn,model_name='K-nearest neighbors')


# All the above was more manual way of parameter tuning and validation
# this can also be accomplished by using GridSearchCV and if needed pipelines
# The difference is that with cross validation we don't preselect validation dataset
# it is repeatadly selected Kfold times from training-validation dataset
# For that reason CV can be computationally expensive especially with large datasets




# PART 2. Cross validation  evaluation of several models with default parameters
print("Starting part2. Testing default parameters for other models using cross_validate......")

models = {"knn": KNeighborsClassifier(), "LogReg": LogisticRegression(random_state=5),
          "DecTree": DecisionTreeClassifier(random_state=5), "RandForest": RandomForestClassifier(random_state=5),
          "HGB":HistGradientBoostingClassifier(random_state=5), "SVC":SVC(random_state=5)}


# Creating variables to store score and performance values
cv_reports = {}
cv_scores = {}
cv_cms = {}
cv_acc = {}
#looping over all models, calculting CV performance and test_scores
fig = plt.figure(2, figsize=BIG)
plt.suptitle("Confusion matrices for default models")
n = 1
for name, model in models.items():
    scoring = ['accuracy','precision_macro','recall_macro']
    kf = KFold(n_splits=5,shuffle=True, random_state=5)
    cv_result = cross_validate(model,X_tv_scaled, y_tv, cv=kf, scoring=scoring, n_jobs=-1)    
    cv_scores[name] = cv_result
    model.fit(X_tv_scaled,y_tv)    
    rep, cm, ac = print_results(model,PLOT,1,name,"_default")
    cv_acc[name] = ac
    cv_cms[name] = cm
    cv_reports[name] = rep
    ax = plt.subplot(2,len(models)//2,n)
    disp = ConfusionMatrixDisplay(cm)    
    disp.plot(ax=ax)
    disp.ax_.set_xlabel("")
    disp.ax_.set_ylabel("")        
    ax.set_title(name)    
    n += 1
plt.tight_layout(pad=1.5,h_pad=3)
plt.savefig("Figure2_confusion_matrices_for_default_models.png")
plt.show()

# Creating dataframe cv_scores
cv_performance_df = pd.DataFrame(cv_scores).transpose()
cv_performance_df_mean = cv_performance_df.applymap(np.mean)
print(cv_performance_df_mean)

if PLOT:
    plt.figure(3, figsize=BIG)
    plt.suptitle("Metrics during cross validation")
    ax = plt.subplot(2,2,1)
    ax.boxplot(cv_performance_df['test_accuracy'], labels=models.keys())
    ax.set_title('Test accuracy ')
    ax.set_ylabel('Test accuracy')
    ax = plt.subplot(2,2,2)
    ax.boxplot(cv_performance_df['test_precision_macro'], labels=models.keys())
    ax.set_title('test precision macro ')
    ax.set_ylabel('test precision macro')
    ax = plt.subplot(2,2,3)
    ax.boxplot(cv_performance_df['test_recall_macro'], labels=models.keys())
    ax.set_title('test recall macro ')
    ax.set_ylabel('test recall macro')
    ax = plt.subplot(2,2,4)
    ax.boxplot(cv_performance_df['fit_time'], labels=models.keys())
    ax.set_title('fit time ')
    ax.set_ylabel('fit time')
    plt.savefig("Figure3_cross_validate_metrics.png")
    plt.show()    

if PLOT:
    plt.figure(4, figsize=BIG)
    plt.suptitle("Mean values of metrics during cross validation")
    ax = plt.subplot(2,2,1)
    ax.plot(models.keys(),cv_performance_df_mean['test_accuracy'].values,marker = 'o', markersize=10)
    ax.set_title('Test accuracy mean')
    ax.set_ylabel('Test accuracy')
    ax = plt.subplot(2,2,2)
    ax.plot(models.keys(),cv_performance_df_mean['test_precision_macro'].values, marker = 'o', markersize=10)
    ax.set_title('test precision macro mean')
    ax.set_ylabel('test precision macro')
    ax = plt.subplot(2,2,3)
    ax.plot(models.keys(),cv_performance_df_mean['test_recall_macro'].values, marker = 'o', markersize=10)
    ax.set_title('test recall macro mean')
    ax.set_ylabel('test recall macro')
    ax = plt.subplot(2,2,4)
    ax.plot(models.keys(),cv_performance_df_mean['fit_time'].values, marker = 'o', markersize=10)
    ax.set_title('fit time mean')
    ax.set_ylabel('fit time')
    plt.savefig("Figure4_mean_of_cross_val_metrics.png")
    plt.show()    





# PART3 GridSearchCV for all the models.
print("Staring part 3: GridSearchCV for tuning all models")

grids = {"knn":{'n_neighbors': range(1,11), 'weights':['uniform','distance']},
         "LogReg": {'solver':['lbfgs','newton-cg'], "C":[0.01, 0.1, 1, 10, 100]}, 
         "DecTree":{"criterion":["gini","entropy"], "max_depth":[5,10,None]}, 
         "RandForest": {"criterion":["gini","entropy"], "max_depth":[5,10,None]},
         "HGB": {"max_depth": [5,10,None], "learning_rate": [0.1, 0.5, 1]},
         "SVC": {"kernel": ["linear", "poly"], "C": [0.01, 0.1, 1, 10, 100]}}

GSCV_reports ={}
GSCV_scores = {}
GSCV_cms = {}
GSCV_best_params = {}
plt.figure(5, figsize=BIG)
plt.suptitle("Confusion matrices for tuned models")
n=1
for name, model in models.items():
    kf = KFold(n_splits=5, shuffle=True, random_state=5)    
    gscv = GridSearchCV(model, grids[name], cv=kf, n_jobs=-1)
    gscv.fit(X_tv_scaled, y_tv)
    print(f'For {name} model best parameters are: {gscv.best_params_} \
           with accuracy score of {gscv.best_score_}')
    
    rep, cm, ac = print_results(model,PLOT,1,name,"_tuned")
    GSCV_best_params[name] = gscv.best_params_
    GSCV_scores[name] = ac
    GSCV_cms[name] = cm
    GSCV_reports[name] = rep
    ax = plt.subplot(2,len(models)//2,n)
    disp = ConfusionMatrixDisplay(cm)  
    disp.plot(ax=ax,cmap='viridis')
    disp.ax_.set_xlabel("")
    disp.ax_.set_ylabel("")             
    ax.set_title(name)    
    n += 1
plt.tight_layout(pad=1.5,h_pad=3)
plt.savefig("Figure5_confusion_matrices_for_tuned_models.png")
plt.show()
    
    
if PLOT:
    plt.figure(6,figsize=SMALL)
    plt.plot(models.keys(), GSCV_scores.values(), marker = 'o', markersize=10)
    plt.ylabel("Accuracy score")
    plt.title("Final test scores after hyperparameter tuning")
    plt.savefig("Figure6_final_test_scores.png")
    plt.show()


   
    
# PART 5. Feature importance and checking subsets of features
    
# With some ML model we can get the evaluation of feture importances
# We will now check most important features according to our decision tree 
# Random forest models

dtc = DecisionTreeClassifier(random_state=5)
dtc.set_params(**GSCV_best_params['DecTree'])
dtc.fit(X_tv_scaled, y_tv)
dt_features = dtc.feature_importances_

rfc = RandomForestClassifier(random_state=5)
rfc.set_params(**GSCV_best_params['RandForest'])
rfc.fit(X_tv_scaled, y_tv)
rf_features = rfc.feature_importances_


# We can see that for both models most important features are at indexes 3 and 5
print(dt_features)
print(rf_features)
print(f'best feature in Decision Tree model is at index {dt_features.argmax()}')
print(f'best feature in Decision Tree model is at index {dt_features.argmax()}')


# I could at this point make some form of Principal component analysis
# or some other tehnique to reduce the number of dimensions (features),
# But here I will here test some of the models with the subset of
# two most important features which we can use for some ploting

X_red_train = X_tv_scaled[:,[3,5]]
y_red_train = y_tv    
X_red_test = X_test_scaled[:,[3,5]]
y_red_test = y_test


# Calculating reports and confusion matrices using only two features
red_reports = {}
red_scores = {}
red_cms = {}
red_acc = {}
plt.figure(7, figsize=BIG)
plt.suptitle("Confusion matrices for reduced models")
n = 1
for name, model in models.items():
    scoring = ['accuracy','precision_macro','recall_macro']
    kf = KFold(n_splits=5,shuffle=True, random_state=5)
    red_result = cross_validate(model,X_tv_scaled, y_tv, cv=kf, scoring=scoring, n_jobs=-1)    
    red_scores[name] = red_result
    model.fit(X_tv_scaled,y_tv)    
    rep, cm, ac = print_results(model,PLOT,1,name,"_reduced")
    red_acc[name] = ac
    red_cms[name] = cm
    red_reports[name] = rep
    ax = plt.subplot(2,len(models)//2,n)
    disp = ConfusionMatrixDisplay(cm)  
    disp.plot(ax=ax, cmap='viridis')
    disp.ax_.set_xlabel("")
    disp.ax_.set_ylabel("")             
    ax.set_title(name)    
    n += 1

plt.tight_layout(pad=1.5,h_pad=3)
plt.savefig("Figure7_confusion_matrices_for_reduced_models.png")
plt.show()


x_min, x_max = X_red_train[:, 0].min() - 0.5, X_red_train[:, 0].max() + 0.5
y_min, y_max = X_red_train[:, 1].min() - 0.5, X_red_train[:, 1].max() + 0.5




# Ploting training and test points for reduced dataset and decision boundary
figure = plt.figure(8,figsize=BIG)
plt.suptitle("Decision boundaries and data points for different models")
n = 1
cm = plt.cm.RdYlBu
cm_points = ListedColormap(["#FF0000","#FFFF00" ,"#0000FF"])


for name, model in models.items():
    ax = plt.subplot(2, len(models)//2, n)
    model.fit(X_red_train,y_red_train)
    score = model.score(X_red_test,y_red_test)
    DecisionBoundaryDisplay.from_estimator(
            model, X_red_train, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )
    # Plot the training points
    ax.scatter(X_red_train[:, 0], X_red_train[:, 1], c=y_red_train, cmap=cm_points, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_red_test[:, 0], X_red_test[:, 1], c=y_red_test, cmap=cm_points, alpha=0.6, edgecolors="k"
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)        
    ax.text(
        x_max - 0.3,
        y_max - 0.5,
        (f'Accuracy: {score:.4f}'),
        size=15,
        horizontalalignment="right",
    )
    n += 1

plt.tight_layout()
plt.savefig("Figure8_decision_maps_2features.png")
plt.show()
    
    
# We see in the last plot that even with only 2 most important features 
# Accuracy is stil very high. Dependin on what the data actually means we 
# discard other features to increase speed and predict with only two features

# If we do want to maximize accuracy and still have a relatively fast model
# For this dataset among tested models K-Nearest Neighbors is best, and I would
# choose it and use it for future unseen data.


# PART 5: Creating selected model 
knn_final = KNeighborsClassifier()
knn_final.set_params(**GSCV_best_params['knn'])
knn.fit(X_tv_scaled,y_tv)
joblib.dump(knn_final, 'final_model.pkl')

    

        
        
