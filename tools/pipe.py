# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:52:49 2024

@author: david
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
class Pipe:
    def __init__(self, 
                 data = {"X_train":None, 
                         "y_train":None, 
                         "X_test":None, 
                         "y_test":None}, 
                 steps = {"preprocessing":dict, 
                          "model":None}):
        
        self.data = data
        self.steps = steps
        self._pipe = None
        self._hyperparameters = {}
                    

    @property
    def pipe(self):
        return self._pipe

    @pipe.setter
    def pipe(self, pipeline):
        self._pipe = pipeline
        
    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hp):
        self._hyperparameters = hp

    def buildPipe(self):
        self.pipe = Pipeline(steps=list(self.steps["preprocessing"].items())+
                                   list(self.steps["model"].items()))
        return self.pipe
    
    def defineHP(self):
        for key, prepro in self.steps["preprocessing"].items():
            hp = prepro._get_param_names()
            self.hyperparameters.update(dict(zip(list(map(lambda x: f"{key}__{x}", hp)),
                                                 len(prepro._get_param_names())*[None])))
        for key, model in self.steps["model"].items():
            hp = model._get_param_names()
            self.hyperparameters.update(dict(zip(list(map(lambda x: f"{key}__{x}", hp)),
                                                 len(model._get_param_names())*[None])))
        
    def fitModel(self):
        searched_params = {key: value for key, value in self.hyperparameters.items() if value is not None}
        self.search = GridSearchCV(self.buildPipe(), searched_params, cv = 5, n_jobs=8)
        self.search.fit(self.data["X_train"], self.data["y_train"])
        print("Best parameter (CV score=%0.3f):" % self.search.best_score_)
        print(self.search.best_params_)
    
    def recap(self):
        return self.search.cv_results_['mean_test_score']