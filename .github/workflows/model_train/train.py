#STEP 1:import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
  f1_score,
  accuracy_score,
  classification_report,
  confusion_matrix
)
#STEP 2: Load Dataset
data=load_breast_cancer()
X=pd.DaraFrame(
  data.data,
  columns=data.feature_names
)
y=pd.Series(data.target,name="target")
print("Dataset shape:"X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\n Trarget values:")
print(y.value_counts())
#STEP 3:Split dataset
X_train,X_test,y_train,y_test=train_test_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  startify=y
)
print("\nTraining samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])
#STEP 4:Define classification algorithms
models={
  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),
  "KNN":KNeighborsClassifier(),
  "Decision Tree"=DecisionTreeClassifier(
  random_state=42
),
"RandomForest:"RandomForestClassifier(
  random_state=42
),
"SVM":SVC()
}
#step5:Define hyperparameter grids
param_grids={
  "Logistic Regression":{
    "model__C":[0,1,1,10],
    "model_solver":["liblinear"]
  },
  "KNN":{
    "model__n_neighbours":[3,5,7,9],
    "model__weights":["uniform","distance"]
  },
  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_samples_split":[2,5,10]
  },
  "RandomForest":{
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
    "model__min_samples_split":[2,5]
  },
  "SVM":{
    "model__C":[0,1,1,10],
    "model__kernel":["linear",rbf"],
    "model__gamma":["scale","auto"]
                     }
}
#STEP6:Run GidSearchCV for every algorithm
results=[]
best_models={}
for model_name,model in models.items():
  print("/n" + "="*60)
  print("Running:",model_name)
  print("="*60)
  pipeline=pipeline([
    ("scaler",StandardScaler())
    ("model",model)
  ])
  grid_search=GridSearchCV(
    estiamtor=pipeline,
    param_grid=param_grids[model_name],
    scoring="f1",
    cv=5,
    n_jobs=-1
  )
  grid_search.fit(X_train,y_train)
  best_models[model_name]=grid_search.best_estimator_results.append({
    "Model":model_name,
    "Best F1 Score":grid_search.best_score_,
    "Best Parameters":grid_search.best_params_
  })
  print("Best f1 Score:",
        round(grid_search.best_score_,4))
  print("Best Parameters:",
        gris_search.best_params_)
#STEP&:Compare all models
results_df=pd.DataFrame(results)
results_df=results_df.
    
