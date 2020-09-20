import abc
from onepiecepredictor.OnePieceRegression import *
from onepiecepredictor.MultiModelsPredictor import MultiModelsPredictor


class MultiModelsRegression(MultiModelsPredictor):
    """
        This class can be used for hyper parameter tuning with cross validation and stratified splitting of data if required.

        X -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        Y -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        testX -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
        testY -> array-like(supported by Sklearn), test data. Ingnored if testTrainSplit is passed
        testTrainSplit -> float, ratio passed will be the amount of test data.
        performCV -> bool, Used when hyperParams not passed to perform plain CV.
        folds -> int, No of folds to be used for CV.
        scoring -> str, Evaluation metric. Currently supported values: r2,neg_mean_squared_error. If not passed r2 is used.
        targetEncodeCols -> List. List of columns to target encode.

    """

    def __init__(self, X, Y, testX = None, testY = None,testTrainSplit = None,
                 folds = None, scoring = None, performCV = None, targetEncodeCols = None):

        super().__init__(X=X, Y=Y, testX=testX, testY=testY, testTrainSplit=testTrainSplit,
                         folds=folds, scoring=scoring, performCV=performCV, targetEncodeCols=targetEncodeCols)

    def predict(self):
        dummyRef = OnePieceRegression(X=self.X, Y=self.Y, model="LINEAR", modelParams={},
                                      testTrainSplit=self.testTrainSplit,
                                      testX=self.testX, testY=self.testY, folds=self.folds, scoring=self.scoring,
                                      performCV=self.performCV)

        tempX = dummyRef.trainX
        tempY = dummyRef.trainY
        tempTestX = dummyRef.testX
        tempTestY = dummyRef.testY

        regressors = ["LINEAR","RF","SVM","KNN","ADABOOST","XGBOOST","CATBOOST"]
        resultsDict = {}
        for regressor in regressors:
            op = OnePieceRegression(X = tempX, Y = tempY, model = regressor, testX = tempTestX, testY = tempTestY, testTrainSplit = None, folds = self.folds,
                                    scoring = self.scoring, performCV = self.performCV, targetEncodeCols = self.targetEncodeCols)

            op.fit()
            score, preds = op.predict()
            resultsDict[regressor] = score

        print(resultsDict)




