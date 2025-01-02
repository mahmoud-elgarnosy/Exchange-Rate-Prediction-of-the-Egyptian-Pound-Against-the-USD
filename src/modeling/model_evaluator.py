from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from src.modeling.base_model import BaseModel
from src.modeling.model_metrics import ModelMetrics


class ModelEvaluator:
    """Class for evaluating and comparing different models."""

    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.models: Dict[str, BaseModel] = {}
        self.metrics: Dict[str, ModelMetrics] = {}

    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a model to the evaluator."""
        self.models[name] = model

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train and evaluate all added models."""
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self.metrics[name] = ModelMetrics(
                rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                mae=mean_absolute_error(y_test, y_pred),
                r2=r2_score(y_test, y_pred),
                feature_importance=model.get_feature_importance(X_test, y_test),
                predictions=y_pred,
                actual_values=y_test.values,
                model_name=name,
                best_params=getattr(model, 'best_params', None)
            )

            print(f"\nMetrics for {name}:")
            print(f"RMSE: {self.metrics[name].rmse:.4f}")
            print(f"MAE: {self.metrics[name].mae:.4f}")
            print(f"R2: {self.metrics[name].r2:.4f}")

            if self.metrics[name].best_params:
                print(f"\nBest parameters for {name}:")
                print(self.metrics[name].best_params)

    def get_best_model(self) -> Tuple[str, ModelMetrics]:
        """Get the best performing model based on RMSE."""
        best_model = min(self.metrics.items(), key=lambda x: x[1].rmse)
        return best_model

    def save_best_model(self, save_dir: str) -> str:
        """
        Save the best performing model to disk.

        Args:
            save_dir: Directory path where the model should be saved

        Returns:
            str: Path to the saved model file
        """
        if not self.metrics:
            raise ValueError("No models have been trained and evaluated yet")

        best_model_name, best_metrics = self.get_best_model()
        best_model = self.models[best_model_name]

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save model with metrics in filename
        filename = f"{best_model_name}_rmse{best_metrics.rmse:.4f}_r2{best_metrics.r2:.4f}.joblib"
        save_path = os.path.join(save_dir, filename)

        joblib.dump(best_model, save_path)
        print(f"Best model ({best_model_name}) saved to: {save_path}")

        return save_path

    def plot_model_comparison(self) -> go.Figure:
        """Plot performance comparison of all models."""
        fig = go.Figure()

        metrics_df = pd.DataFrame({
            'Model': list(self.metrics.keys()),
            'RMSE': [m.rmse for m in self.metrics.values()],
            'MAE': [m.mae for m in self.metrics.values()],
            'R2': [m.r2 for m in self.metrics.values()]
        })

        for metric in ['RMSE', 'MAE', 'R2']:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(4),
                textposition='auto',
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            xaxis_title='Model',
            yaxis_title='Score',
            showlegend=True
        )

        return fig

    def plot_predictions(self) -> go.Figure:
        """Plot actual vs predicted values for all models with their corresponding actual values."""
        fig = go.Figure()
        colors = ['red', 'blue', 'green', 'purple', 'orange']

        for idx, (model_name, metrics) in enumerate(self.metrics.items()):
            fig.add_trace(go.Scatter(
                y=metrics.actual_values,
                name=f'Actual ({model_name})',
                mode='lines',
                line=dict(color='black', width=2, dash='solid'),
                showlegend=idx == 0
            ))

            fig.add_trace(go.Scatter(
                y=metrics.predictions,
                name=f'{model_name} Predictions',
                mode='lines',
                line=dict(
                    color=colors[idx % len(colors)],
                    width=1.5,
                    dash='dot'
                ),
                customdata=[[metrics.rmse, metrics.mae, metrics.r2]],
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Predicted: %{y:.4f}<br>"
                    "RMSE: %{customdata[0][0]:.4f}<br>"
                    "MAE: %{customdata[0][1]:.4f}<br>"
                    "R²: %{customdata[0][2]:.4f}<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title='Model Predictions Comparison',
            xaxis_title='Sample Index',
            yaxis_title='Exchange Rate Value',
            plot_bgcolor='white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)'),
            hovermode='x unified'
        )

        return fig
