from AlgorithmImports import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


# a method to load the model for the crypto metadata strategy
def load_model(model_name: str):
    '''Loads the model. 
    
    Arguments: 
        model_name: The model name. 
    
    Returns: The model object. 
    '''
    if model_name not in {
        'rf', 'lr', 'svm', 'gb', 'knn', 'stacking', 'votinghard', 'votingsoft'
    }:
        raise ValueError(f'Unsupported model name {model_name}!')

    ### set hyperparameters here
    # for rf
    max_depth = 7 # [2, 5, 7]
    # for svm
    C = 1e-2 # [1e-4, 1e-2, 1, 10]
    # for gb
    learning_rate = 1e-4 # [1e-4, 1e-3, 1e-2]
    n_estimators = 150 # [100, 150, 250]
    # for knn
    n_neighbours = 3 # [3, 5, 7]
    
    if model_name == 'rf':
        # set hyperparameters
        model =  make_pipeline(
            StandardScaler(), 
            RandomForestClassifier(random_state=4801, max_depth=max_depth)
        )
    elif model_name == 'lr':
        model =  make_pipeline(
            StandardScaler(), 
            LogisticRegressionCV(random_state=4801, Cs=[1e-4, 1e-2, 1, 10], n_jobs=-1)
        )
    elif model_name == 'svm':
        model = make_pipeline(
            StandardScaler(), 
            SVC(random_state=4801, probability=True, C=C)
        )
    elif model_name == 'gb':
        model = make_pipeline(
            StandardScaler(), 
            GradientBoostingClassifier(random_state=4801, learning_rate=learning_rate, n_estimators=n_estimators)
        )
    elif model_name == 'knn':
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbours)
        )
    elif model_name == 'stacking':
        # set individual hyperparameters first
        model = make_pipeline(
            StandardScaler(),
            StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=4801, max_depth=max_depth)),
                    ('lr', LogisticRegressionCV(random_state=4801, Cs=[1e-4, 1e-2, 1, 10], n_jobs=-1)),
                    ('svm', SVC(random_state=4801, probability=True, C=C)),
                    ('gb', GradientBoostingClassifier(random_state=4801, learning_rate=learning_rate, n_estimators=n_estimators)),
                    ('knn', KNeighborsClassifier(n_neighbors=n_neighbours))
                ],
                n_jobs=-1
            )
        )
    elif model_name == 'votinghard':
         model = make_pipeline(
            StandardScaler(),
            VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=4801, max_depth=max_depth)),
                    ('lr', LogisticRegressionCV(random_state=4801, Cs=[1e-4, 1e-2, 1, 10], n_jobs=-1)),
                    ('svm', SVC(random_state=4801, probability=True, C=C)),
                    ('gb', GradientBoostingClassifier(random_state=4801, learning_rate=learning_rate, n_estimators=n_estimators)),
                    ('knn', KNeighborsClassifier(n_neighbors=n_neighbours))
                ],
                voting='hard',
                n_jobs=-1
            )
        )
    elif model_name == 'votingsoft':
         model = make_pipeline(
            StandardScaler(),
            VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=4801, max_depth=max_depth)),
                    ('lr', LogisticRegressionCV(random_state=4801, Cs=[1e-4, 1e-2, 1, 10], n_jobs=-1)),
                    ('svm', SVC(random_state=4801, probability=True, C=C)),
                    ('gb', GradientBoostingClassifier(random_state=4801, learning_rate=learning_rate, n_estimators=n_estimators)),
                    ('knn', KNeighborsClassifier(n_neighbors=n_neighbours))
                ],
                voting='soft',
                n_jobs=-1
            )
        )
    return model

