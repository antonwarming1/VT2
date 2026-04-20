import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import requests
from io import StringIO
import urllib3
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import optuna
from scipy.stats import randint
# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Download the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
response = requests.get(url, verify=False)  # Bypass SSL verification
data = StringIO(response.text)

# Read the data into a pandas DataFrame
df = pd.read_csv(data, sep=r"\s+", header=None)

# Assign column names
column_names = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 
    'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 
    'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 
    'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 
    'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 
    'Dirtiness', 'Bumps', 'Other_Faults'
]

df.columns = column_names


# Define fault categories
surface_faults = ['Stains', 'Dirtiness',"Pastry"]
other_faults = ['Z_Scratch', 'K_Scatch', 'Bumps', 'Other_Faults']



print(f"\nTotal Surface Faults (Stains + Dirtiness): {df[surface_faults].sum().sum()}")
print(f"Total Other Faults: {df[other_faults].sum().sum()}")

# Binary target: 1 = Stains/Dirtiness/pastry, 0 = Other faults
df['target'] = np.where(df[surface_faults].any(axis=1), 1, 0)

# Drop the individual fault columns
df = df.drop(columns=surface_faults + other_faults)

# Data Exploration
print(df.head(10))
#print(df.info())
#print(df.describe())

# Check for missing values
print(df.isnull().sum())
#no missing values found

# Visualize the distribution
plt.figure(figsize=(10, 5))

# 1) Fix the order of the bars explicitly
order = [0, 1]   # or sorted(df['target'].unique())
sns.countplot(x='target', data=df, order=order)
plt.title('Binary Fault Distribution (1=Stains, Dirtiness and Pastry, 0=Other)')

# 2) Get counts in the SAME order
counts = df['target'].value_counts().reindex(order)

for i, target in enumerate(order):
    count = counts[target]
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)

plt.show()

print("\nClass Distribution:")
print(df['target'].value_counts())

# Prepare features and target variable
X = df.drop('target', axis=1)
y = df['target']

# duo to very imbalanced dataset, we use stratify

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_base_models():
    #Train base Decision Tree and Random Forest models with default parameters. Only used for test and quick comparison.
    dt_model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    #fit models
    dt_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    
    return dt_model, rf_model

#KIG HER FOR GRID SEARCH, RANDOM SEARCH OG BAYESIAN OPTIMIZATION

def grid_search_cv(model_type='tree', scoring='accuracy'):
    # Perform Grid Search CV for hyperparameter tuning of Decision Tree or Random Forest
    if model_type == 'tree':
        # Define parameter grid for Decision Tree
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        #setup model
        model = DecisionTreeClassifier(random_state=42)
    else:  # Random Forest
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [3, 5, 7, 10,None],
            'class_weight': ['balanced']
            
        }
        #setup model
        model = RandomForestClassifier(random_state=42)
    
    # Use StratifiedKFold duo to imbalanced data
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # do the grid search with stratified kfold
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skf,
        n_jobs=-1,
        scoring=scoring
    )
    # fit the grid search
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def random_search_cv(model_type='tree',scoring='accuracy'):
    # Perform Randomized Search CV for hyperparameter tuning of Decision Tree or Random Forest
    
    if model_type == 'tree':
        # Define parameter distributions for Decision Tree
        param_distributions = {
            'criterion': ['gini', 'entropy'],
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'class_weight': ['balanced']
        }
        #setup model
        model = DecisionTreeClassifier(random_state=42)
    else:  
        # Define parameter for Random Forest
        param_distributions = {
            'n_estimators': randint(50, 300),
            'criterion': ['gini', 'entropy'],
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': randint(3, 10),
            'class_weight': ['balanced']
        }
        #setup model
        model = RandomForestClassifier(random_state=42)
    
    # Use StratifiedKFold duo to imbalanced data
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # do the random search with stratified kfold
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=50,
        cv=skf,
        n_jobs=-1,
        scoring=scoring,
        random_state=42
    )
    # fit the random search
    random_search.fit(x_train, y_train)
    
    return random_search.best_params_, random_search.best_score_

def objective(trial, model_type='tree',scoring='accuracy'):
    # Objective function for Optuna Bayesian Optimization
    if model_type == 'tree':
        # Define hyperparameter search space for Decision Tree
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced'])
        }
        #setup model
        model = DecisionTreeClassifier(random_state=42, **params)
    else:  # Random Forest
        # Define hyperparameter search space for Random Forest
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_int('max_features', 3, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced'])
        }
        #setup model
        model = RandomForestClassifier(random_state=42, **params)
    
    # Use StratifiedKFold for imbalanced data
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # get cross-validation score
    score = cross_val_score(model, x_train, y_train, cv=skf, scoring=scoring)
    # define mean score to maximize
    return score.mean()

def bayesian_optimization(model_type='tree'):
    # Perform Bayesian Optimization for hyperparameter tuning using Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # create study and set direction to maximize accuracy
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, model_type), n_trials=50)
    
    return study.best_params, study.best_value

def plot_cv_scores(best_score_grid, best_score_random, best_score_bayes, basic_cv_mean, model_name='Model'):
    # Plot comparison of CV scores from different hyperparameter tuning methods
    methods = ['Grid Search', 'Random Search', 'Bayesian Optimization', 'Base Model']
    # scores list of the methods
    scores = [best_score_grid, best_score_random, best_score_bayes, basic_cv_mean]
    # plot bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x=methods, y=scores)
    plt.ylim(0, 1.1)
    plt.ylabel('Best CV Accuracy')
    plt.title(f'{model_name}: Comparison of Hyperparameter Tuning Methods')
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f"{score:.4f}", ha='center')
    plt.show()



def evaluate_model_metrics( y_true, y_pred):
    
    
    print("\n=== MODEL EVALUATION METRICS ===")
    
    # Accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Detailed metrics per class (precision, recall, f1-score, support)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    print(f"\nClass 0 (Other Faults):")
    print(f"  Precision: {precision[0]:.4f}")
    print(f"  Recall: {recall[0]:.4f}")
    print(f"  F1-Score: {f1[0]:.4f}")
    print(f"  Support: {support[0]}")
    
    print(f"\nClass 1 (Stains, Dirtiness and Pastry):")
    print(f"  Precision: {precision[1]:.4f}")
    print(f"  Recall: {recall[1]:.4f}")
    print(f"  F1-Score: {f1[1]:.4f}")
    print(f"  Support: {support[1]}")    
    
    return accuracy , f1

def plot_confusion_matrix(y_true, y_pred, title):
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Other Faults', 'Stains/Dirtiness/Pastry'],
                yticklabels=['Other Faults', 'Stains/Dirtiness/Pastry'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_confusion_matrix_normalized(y_true, y_pred, title):
    # Plot normalized confusion matrix
    cm_normalized=confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Other Faults', 'Stains/Dirtiness/Pastry'],
                yticklabels=['Other Faults', 'Stains/Dirtiness/Pastry'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_tree_model(model, feature_names):
    # Plot decision tree visualization for Decision Tree model
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['Other Faults', 'Stains, Dirtiness and Pastry'], filled=True)
    plt.title('Decision Tree Visualization')
    plt.show()



def train_model(best_params, model_type='tree'):
    """Train model with best parameters"""
    if model_type == 'tree':
        model = DecisionTreeClassifier(random_state=42, **best_params)
    else:
        model = RandomForestClassifier(random_state=42, **best_params)
    # fit the model
    model.fit(x_train, y_train)
    return model

def evaluate_models(model_type='tree', model_name='Decision Tree'):
    """Run complete evaluation pipeline for a given model type"""
    print("\n" + "="*80)
    print(f"{model_name.upper()} EVALUATION")
    print("="*80)
    
    print("\n" + "="*60)
    print("STEP 1: Evaluating Base Model")
    print("="*60)
    
    # setup Base model with default parameters
    if model_type == 'tree':
        base_model = DecisionTreeClassifier(random_state=42)
    else:
        base_model = RandomForestClassifier(random_state=42)
    # Cross-validation scores for base model with stratified kfold
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # Get cross-validation scores
    basic_cv_scores = cross_val_score(base_model, x_train, y_train, cv=skf, scoring='accuracy')
    # Get mean score
    basic_cv_mean = basic_cv_scores.mean()
    print(f"Base Model CV Scores: {basic_cv_scores}")
    print(f"Base Model Mean CV Accuracy: {basic_cv_mean:.4f}")
    print(f"Base Model Std CV Accuracy: {basic_cv_scores.std():.4f}")
    
    print("\n" + "="*60)
    print("STEP 2: Hyperparameter Tuning")
    print("="*60)
    
    # Get best parameters and scores from different methods
    print("\nRunning Grid Search...")
    best_params_grid, best_score_grid = grid_search_cv(model_type)
    print(f"Grid Search Best Params: {best_params_grid}")
    print(f"Grid Search Best CV Accuracy: {best_score_grid:.4f}")
    
    # Random Search
    print("\nRunning Random Search...")
    best_params_random, best_score_random = random_search_cv(model_type)
    print(f"Random Search Best Params: {best_params_random}")
    print(f"Random Search Best CV Accuracy: {best_score_random:.4f}")
    
    # Bayesian Optimization
    print("\nRunning Bayesian Optimization...")
    best_params_bayes, best_score_bayes = bayesian_optimization(model_type)
    print(f"Bayesian Optimization Best Params: {best_params_bayes}")
    print(f"Bayesian Optimization Best CV Accuracy: {best_score_bayes:.4f}")
    
    print("\n" + "="*60)
    print("STEP 3: Comparing All Methods")
    print("="*60)
    
    # Plot comparison of CV scores
    plot_cv_scores(best_score_grid, best_score_random, best_score_bayes, basic_cv_mean, model_name)
    
    # Find the best method
    methods = {
        'Grid Search': (best_params_grid, best_score_grid),
        'Random Search': (best_params_random, best_score_random),
        'Bayesian Optimization': (best_params_bayes, best_score_bayes),
        'base model': (None, basic_cv_mean)
    }
    # Get the method with the highest score
    best_method = max(methods.items(), key=lambda x: x[1][1])
    best_method_name = best_method[0]
    best_params = best_method[1][0]
    best_score = best_method[1][1]
    
    print(f" Best search {best_method_name}")
    print(f"Best CV Accuracy: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    print("\n" + "="*60)
    print("STEP 4: Final Model Evaluation on Test Set")
    print("="*60)
    
    # Diagnose performance with best parameters if base model not best
    if best_params is None:
        print("Using base model (default parameters)")
        best_params = {}
        method_label = "Base Model"
    else:
        # Use best parameters from the best search method
        print(f"Using {best_method_name} parameters")
        method_label = best_method_name
    
    # Train model
    final_model = train_model(best_params, model_type)
    
    # Get predictions
    y_pred = final_model.predict(x_test)
    
    # Evaluate model
    test_accuracy, test_f1 = evaluate_model_metrics( y_test, y_pred)
    
    # Plot confusion matrix with method name in title
    plot_confusion_matrix(y_test, y_pred,title='Confusion Matrix - '+method_label)
    plot_confusion_matrix_normalized(y_test, y_pred,title='Confusion Matrix Normalized - '+method_label)
    # Plot tree visualization (only for decision tree)
    if model_type == 'tree':
        plot_tree_model(final_model, feature_names=x_train.columns)
    return test_accuracy, test_f1, best_params, best_method_name

def main():
    # Evaluate Decision Tree
    dt_test_acc, dt_test_f1, dt_params, dt_method = evaluate_models('tree', 'Decision Tree')
    
    # Evaluate Random Forest
    rf_test_acc, rf_test_f1, rf_params, rf_method = evaluate_models('random forest', 'Random Forest')
    
    # Final Comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON BETWEEN DECISION TREE AND RANDOM FOREST")
    print("="*80)
    
    print(f"\nDecision Tree:")
    print(f"  Best Method: {dt_method}")
    print(f"  Test Set Accuracy: {dt_test_acc}")
    print(f"  Test Set F1-Score: {dt_test_f1}")
    print(f"  Best Parameters: {dt_params}")
    
    print(f"\nRandom Forest:")
    print(f"  Best Method: {rf_method}")
    print(f"  Test Set Accuracy: {rf_test_acc}")
    print(f"  Test Set F1-Score: {rf_test_f1}")
    print(f"  Best Parameters: {rf_params}")
    # Determine which model performed better
    if dt_test_acc > rf_test_acc:
        best_model_comparison = "Decision Tree"
        margin = dt_test_acc - rf_test_acc
    else:
        best_model_comparison = "Random Forest"
        margin = rf_test_acc - dt_test_acc
    
    print(f" {best_model_comparison} performed better on the test set.")
    print(f"  {margin} better, and ({margin*100:.2f}% better)")
    
    # Side-by-side comparison plot of performance
    plt.figure(figsize=(10, 5))
    models = ['Decision Tree', 'Random Forest']
    scores = [dt_test_acc, rf_test_acc]
    colors = ["blue", "red"]
    sns.barplot(x=models, y=scores, palette=colors)
    plt.ylim(0, 1)
    plt.ylabel('Test Set Accuracy')
    plt.title('Decision Tree vs Random Forest - Test Set Performance')
    for  i, score in enumerate(scores):
        plt.text(i, score + 0.01, f"{score:.4f}", ha='center', fontweight='bold')
    plt.show()


if __name__ == "__main__":
    main()


    # Simple training and evaluation of base models. Can be used for quick comparison and finding tuning ranges.
    """
    dt_model, rf_model = train_base_models()
    #compar the models metrics

    y_pred_dt = dt_model.predict(x_test)
    y_pred_rf = rf_model.predict(x_test)
    test_accuracy_dt, f1_dt = evaluate_model_metrics(dt_model, y_test, y_pred_dt)
    test_accuracy_rf, f1_rf = evaluate_model_metrics(rf_model, y_test, y_pred_rf)
    print(f"\nDecision Tree Test Accuracy: {test_accuracy_dt}")
    print(f"Random Forest Test Accuracy: {test_accuracy_rf}")
    print(f"\nDecision Tree F1-Score: {f1_dt}")
    print(f"Random Forest F1-Score: {f1_rf}")
    """
    

   


