from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from src.modeling.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient Boosting implementation with grid search."""

    def __init__(self):
        super().__init__()
        self.param_grid = {
            'max_depth': [-1, 3, 5, 7],
            'learning_rate': [.001, .005, 0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200, 300, 400],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4]
        }

    def _create_model(self) -> None:
        params = {
            "n_estimators": 700,
            "max_depth": 4,
            "min_samples_split": 5,
            "learning_rate": 0.01,
            "loss": "squared_error",
        }
        self.model = GradientBoostingRegressor(**params, random_state=42)

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        importance_scores = self.model.feature_importances_
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        return importance.sort_values('importance', ascending=False)


class RandomForestModel(BaseModel):
    """Random Forest implementation with grid search."""

    def __init__(self):
        super().__init__()
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def _create_model(self) -> None:
        base_model = RandomForestRegressor(random_state=42)
        self.model = GridSearchCV(
            base_model,
            self.param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.best_estimator_.feature_importances_
        })
        self.best_params = self.model.best_params_
        return importance.sort_values('importance', ascending=False)


class SVMModel(BaseModel):
    """Support Vector Machine implementation with grid search."""

    def __init__(self):
        super().__init__()
        self.param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.2, 0.5]
        }

    def _create_model(self) -> None:
        base_model = SVR()
        self.model = GridSearchCV(
            base_model,
            self.param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.model.best_estimator_.kernel == 'linear':
            importance_scores = np.abs(self.model.best_estimator_.coef_[0])
        else:
            X_scaled = self.scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            baseline_predictions = self.model.predict(X_scaled_df)
            baseline_score = r2_score(y, baseline_predictions)
            importance_scores = []

            for col in self.feature_names:
                X_permuted = X_scaled_df.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col])
                permuted_predictions = self.model.predict(X_permuted)
                permuted_score = r2_score(y, permuted_predictions)
                importance_scores.append(baseline_score - permuted_score)

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        self.best_params = self.model.best_params_
        return importance.sort_values('importance', ascending=False)


class LinearRegressionModel(BaseModel):
    """Linear Regression implementation."""

    def __init__(self):
        super().__init__()

    def _create_model(self) -> None:
        self.model = LinearRegression()

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.model.coef_)
        })
        return importance.sort_values('importance', ascending=False)
