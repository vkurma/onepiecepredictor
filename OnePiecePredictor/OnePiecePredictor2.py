from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
import gc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
import abc
from category_encoders import TargetEncoder
from imblearn.under_sampling import RandomUnderSampler

###################################################################################
## Support Undersampling as well.
## Return Predictions from train as well.
###################################################################################
class OnePiecePredictor2(metaclass=abc.ABCMeta):

    def __init__(self, X, Y, model, modelParams = {}, testX = None, testY = None,testTrainSplit = None,
                 folds = None, hyperParams = None, scoring = None, performCV = None, targetEncodeCols = None):

        self.X = X
        self.Y = Y
        self.testX = testX
        self.testY = testY
        self.testTrainSplit = testTrainSplit
        self.folds = folds
        self.model = model
        self.hyperParams = hyperParams
        self.bestEstimator = None
        self.scoring = scoring
        self.targetEncodeCols = targetEncodeCols
        self.performCV = performCV
        self.modelParams = modelParams
        self.estimatorModel = self.getEstimatorModel()
        self.trainX, self.testX, self.trainY, self.testY = self.getTestTrainSlipt()

    @abc.abstractmethod
    def getTestTrainSlipt(self):
        pass


    def fit(self):
        if(self.hyperParams):
            self.__trainWithGridCV()
        elif(self.performCV):
            self.__trainWithCV()
        else:
            self.__trainNoCV()

    def __trainNoCV(self):
        self.estimatorModel.fit(self.trainX, self.trainY)
        self.bestEstimator = self.estimatorModel

    def __trainWithCV(self):
        stratKfold = StratifiedKFold(n_splits=self.folds)
        crossValScores = cross_val_score(self.estimatorModel, self.testX, y=list(self.testY), scoring=self.scoring, cv=stratKfold)
        print("Cross Validation Scores")
        print(crossValScores)
        self.bestEstimator = self.estimatorModel

    def __trainWithGridCV(self):
        gridSearch = GridSearchCV(self.estimatorModel, self.hyperParams, cv=self.folds, scoring=self.scoring)
        gridSearch.fit(self.trainX, self.trainY)
        cvres = gridSearch.cv_results_
        for meanTestScore, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(meanTestScore, params)
        self.bestEstimator = gridSearch.best_estimator_

    def test(self):
        preds = self.bestEstimator.predict(self.testX)
        return preds

    @abc.abstractmethod
    def predict(self):
        pass

    def newDataPredict(self, testData):
        preds = self.bestEstimator.predict(testData)
        return preds

    @abc.abstractmethod
    def getEstimatorModel(self):
        pass