from OnePiecePredictor import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import sklearn
from sklearn.metrics import *
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from .OnePiecePredictor import OnePiecePredictor

class OnePieceRegression(OnePiecePredictor):
    """
        This class can be used for hyper parameter tuning with cross validation and stratified splitting of data if required.

        X -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        Y -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        model -> string Currently supported models: LOGISTIC,RF,SVM,KNN,ADABOOST,XGBOOST,CATBOOST
        testX -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
        testY -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
        testTrainSplit -> float, ratio passed will be the amount of test data.
        hyperParams -> dictionary, Hyper parameters specific to the model passed. If passed CV is performed.
        performCV -> bool, Used when hyperParams not passed to perform plain CV.
        folds -> int, No of folds to be used for CV.
        scoring -> str, Evaluation metric. Currently supported values: r2,neg_mean_squared_error. If not passed r2 is used.
        targetEncodeCols -> List. List of columns to target encode.
        modelParams -> dictionary, Any model specific parameters can be passed as dictionary.

    """

    def __init__(self, X, Y, model, modelParams = {}, testX = None, testY = None,testTrainSplit = None,
                 folds = None, hyperParams = None,scoring = None, performCV = None, targetEncodeCols = None):
        super().__init__(
            X = X, Y = Y, testX=testX, testY=testY, testTrainSplit=testTrainSplit,
            model=model, folds=folds, hyperParams=hyperParams,scoring=scoring, performCV=performCV, targetEncodeCols = targetEncodeCols,
        modelParams = modelParams)
        self.scoreToFuncDict = self.__getScoreToFuncDict()
        if(not self.scoring):
            self.scoring = 'r2'


    def getTestTrainSlipt(self):
        ## If both testX and testTrainSplit are not passed throw exception.
        if ((self.testX is None) and (self.testTrainSplit is None)):
            raise Exception("Please pass testX or testTrainSplit")

        if (self.targetEncodeCols):
            for col in self.targetEncodeCols:
                encoder = TargetEncoder()
                self.X[col] = encoder.fit_transform(self.X[col])
                if (self.testX):
                    self.testX[col] = encoder.fit_transform(self.testX[col])

        if(self.testTrainSplit):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=self.testTrainSplit, random_state=7)
            return X_train, X_test, y_train, y_test
        else:
            return self.X, self.testX, self.Y, self.testY


    def fit(self):
        super().fit()

    def test(self):
        super().test()

    def predict(self):
        if (self.testY is None):
            return 0, self.test()

        preds = self.bestEstimator.predict(self.testX)
        res = getattr(sklearn.metrics, self.scoreToFuncDict[self.scoring])(self.testY, preds)
        print(res)
        return res, preds

    def newDataPredict(self, testData):
        super().newDataPredict()

    def getEstimatorModel(self):

        modelToClassMapper = {
            "LINEAR" : LinearRegression(),
            "RF" : RandomForestRegressor(),
            "SVM" : SVR(),
            "KNN" : KNeighborsRegressor(),
            "ADABOOST" : AdaBoostRegressor(),
            "XGBOOST" : XGBRegressor(),
            "CATBOOST" : CatBoostRegressor(),
        }

        if(self.model not in modelToClassMapper):
            raise Exception("Please pass a valid model from: LINEAR,RF,SVM,KNN,ADABOOST,XGBOOST,CATBOOST")
        else:
            cls = modelToClassMapper[self.model]
            if ((self.model == "CATBOOST") and ('logging_level' not in self.modelParams)):
                self.modelParams['logging_level'] = 'Silent'

            return cls.set_params(**self.modelParams)

    def __getScoreToFuncDict(self):

        scoreToFuncDict = {
            "r2": 'r2_score',
            "neg_mean_squared_error" : 'mean_squared_error'
        }

        return scoreToFuncDict


