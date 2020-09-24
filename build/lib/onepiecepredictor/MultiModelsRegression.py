import abc
from onepiecepredictor.OnePieceRegression import OnePieceRegression
from onepiecepredictor.MultiModelsPredictor import MultiModelsPredictor


class MultiModelsRegression(MultiModelsPredictor):
    """
        This class can be used for hyper parameter tuning with cross validation and stratified splitting of data if required.

        X -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        Y -> array-like(supported by Sklearn). If testTrainSplit is passed, this will be split into train and test
        testX -> array-like(supported by Sklearn), test data. Ignored if testTrainSplit is passed
        testY -> array-like(supported by Sklearn), test data. Ignored if testTrainSplit is passed
        testTrainSplit -> float, ratio passed will be the amount of test data.
        performCV -> bool, Used when hyperParams not passed to perform plain CV.
        folds -> int, No of folds to be used for CV.
        scoring -> str, Evaluation metric. Currently supported values: r2,neg_mean_squared_error. If not passed r2 is used.
        targetEncodeCols -> List. List of columns to target encode.

    """

    def __init__(self, X, Y, testX = None, testY = None,testTrainSplit = None,
                 folds = 5, scoring = None, performCV = None, targetEncodeCols = None):
        super().__init__(X=X, Y=Y, testX=testX, testY=testY, testTrainSplit=testTrainSplit,
                         folds=folds, scoring=scoring, performCV=performCV, targetEncodeCols=targetEncodeCols)

    """
    Returns dictionary with keys as Models and Values as metric scores.
    """
    def predict(self):
        dummyRef = OnePieceRegression(X=self.X, Y=self.Y, model="LINEAR", modelParams=None,
                                      testTrainSplit=self.testTrainSplit,
                                      testX=self.testX, testY=self.testY, folds=self.folds, scoring=self.scoring,
                                      performCV=self.performCV, targetEncodeCols= self.targetEncodeCols)

        tempX = dummyRef.trainX
        tempY = dummyRef.trainY
        tempTestX = dummyRef.testX
        tempTestY = dummyRef.testY

        regressors = ["LINEAR","RF","SVM","KNN","ADABOOST","XGBOOST","CATBOOST"]
        resultsDict = {}
        for regressor in regressors:
            op = OnePieceRegression(X = tempX, Y = tempY, model = regressor, modelParams = {},testX = tempTestX, testY = tempTestY, testTrainSplit = None, folds = self.folds,
                                    scoring = self.scoring, performCV = self.performCV, targetEncodeCols = None)

            op.fit()
            score, preds = op.predict()
            resultsDict[regressor] = score

            del op

        return resultsDict




