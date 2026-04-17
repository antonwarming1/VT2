class
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
