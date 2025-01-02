from typing import Tuple
from src.modeling.base_model import BaseModel
from src.modeling.model_metrics import ModelMetrics
from src.modeling.models import RandomForestModel, GradientBoostingModel, SVMModel, LinearRegressionModel
from src.modeling.model_evaluator import ModelEvaluator
import pandas as pd


def run_model_comparison(X: pd.DataFrame, y: pd.Series) -> Tuple[str, BaseModel, ModelMetrics]:
    """Main function to run the entire model comparison process."""
    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Add models
    evaluator.add_model('Random Forest', RandomForestModel())
    evaluator.add_model('XGBoost', GradientBoostingModel())
    evaluator.add_model('SVM', SVMModel())
    evaluator.add_model('Linear Regression', LinearRegressionModel())

    # Train and evaluate models
    evaluator.train_and_evaluate(X, y)

    # Get best model
    best_model_name, best_metrics = evaluator.get_best_model()
    evaluator.save_best_model("../models")

    # Plot comparisons
    comparison_plot = evaluator.plot_model_comparison()
    comparison_plot.show()

    # Plot best model predictions
    predictions_plot = evaluator.plot_predictions()
    predictions_plot.show()

    print(f"\nBest Model: {best_model_name}")
    print(f"RMSE: {best_metrics.rmse:.4f}")
    print(f"MAE: {best_metrics.mae:.4f}")
    print(f"R2: {best_metrics.r2:.4f}")

    if best_metrics.best_params:
        print("\nBest Parameters:")
        print(best_metrics.best_params)

    print("\nTop 10 Most Important Features:")
    print(best_metrics.feature_importance.head(10))

    return best_model_name, evaluator.models[best_model_name], best_metrics
