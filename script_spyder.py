# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:08:06 2024

@author: david
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:35:19 2024

@author: david
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from tools.customPMC import CustomEstimator
from tools.labelToDigits import labelToDigits
from tools.pipe import Pipe
from pathlib import Path
import os
import keras

path_variable = Path(os.getcwd())
train_data = path_variable / "data/train.csv"
test_data = path_variable / "data/test.csv"

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)

full_data = pd.concat([df_train,df_test])

X_train, X_test, y_train, y_test = train_test_split(full_data.iloc[:,:-1], full_data['Activity'], test_size=0.3)
pca = PCA(n_components = 10)
X_train_pca = pca.fit_transform(X_train)
# Apply the same transformation to X_test
X_test_pca = pca.transform(X_test)

y_train_digits = labelToDigits(y_train)
y_test_digits = labelToDigits(y_test)


pipe = Pipe(data = {"X_train":X_train_pca, 
                    "y_train":keras.utils.to_categorical(y_train_digits,6), 
                    "X_test":X_test_pca, 
                    "y_test":keras.utils.to_categorical(y_test_digits,6)}, 
            steps = {"preprocessing":{}, 
                    "model":{'model':CustomEstimator()}})
pipe.defineHP()
pipe.hyperparameters["model__verbose"] = [3]
pipe.fitModel()






# def main():
#     pca = PCA(n_components = 10)
#     X_train_pca = pca.fit_transform(X_train)
#     # Apply the same transformation to X_test
#     X_test_pca = pca.transform(X_test)

#     # pipe = Pipe(data = {"X_train":X_train_pca, 
#     #                     "y_train":y_train, 
#     #                     "X_test":X_test_pca, 
#     #                     "y_test":y_test}, 
#     #             steps = {"preprocessing":{}, 
#     #                     "model":{'model':CustomEstimator()}})
#     # pipe.defineHP()
#     # pipe.hyperparameters['model__activation'] = ['relu']
#     # # pipe.hyperparameters['output_dim'] = [6]
#     # pipe.fitModel()
#     pmc = CustomEstimator()
#     pmc.fit(X_train_pca,y_train)
#     pmc.score(X_test_pca,y_train)
    


# if __name__=='__main__':
#     main()