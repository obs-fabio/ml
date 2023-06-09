import ml.models.base as ml

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight

class RandomForest(ml.Base):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.criterion = kwargs.get('criterion', "gini")
        self.max_depth = kwargs.get('max_depth', None)
        self.max_features = kwargs.get('max_features', None)
        self.random_state = kwargs.get('random_state', 42)
        self.model = None

    def fit(self, X, Y, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            criterion=self.criterion,
            max_features=self.max_features,
            class_weight=self.get_class_weight(Y)
        )
        self.model.fit(X, Y)

    def predict(self, X, output_as_classifier=True, **kwargs):
        if self.model is None:
            raise UnboundLocalError("It is not possible to predict the data without training")
        
        predictions = self.model.predict(X)
        if output_as_classifier:
            predictions = predictions.astype(int)
        return predictions
