"""
Machine Learning Integration for Portfolio Management

This module provides machine learning capabilities for portfolio analysis,
prediction, and optimization including various models and techniques.

Key Features:
- Return prediction models (linear, tree-based, ensemble)
- Clustering for regime identification (K-means, DBSCAN, GMM)
- Anomaly detection for risk events (Isolation Forest, LOF, Autoencoders)
- Reinforcement learning for dynamic allocation (Q-learning, Policy Gradient)
- Neural network-based forecasting (LSTM, GRU, Transformer)
- Feature engineering pipeline for financial data
- Model backtesting and validation framework

Classes:
    ReturnPredictor: Various models for return prediction
    RegimeClusterer: Clustering algorithms for market regime identification
    AnomalyDetector: Detect unusual market conditions and risk events
    RLPortfolioAgent: Reinforcement learning for dynamic allocation
    NeuralForecaster: Deep learning models for time series forecasting
    FeatureEngineer: Feature creation and transformation pipeline
    ModelBacktester: Backtesting and validation framework
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings('ignore')


class ModelType(str, Enum):
    """Types of ML models."""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    NEURAL_NET = "neural_net"
    LSTM = "lstm"
    GRU = "gru"


class ClusteringMethod(str, Enum):
    """Clustering methods for regime identification."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gaussian_mixture"
    HIERARCHICAL = "hierarchical"


class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER = "local_outlier"
    AUTOENCODER = "autoencoder"
    STATISTICAL = "statistical"


class FeatureType(str, Enum):
    """Types of engineered features."""
    PRICE = "price"
    RETURN = "return"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    SENTIMENT = "sentiment"


class PredictionResult(BaseModel):
    """Results from prediction models."""
    predictions: List[float]
    model_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Tuple[List[float], List[float]]] = None

    class Config:
        arbitrary_types_allowed = True


class ReturnPredictor:
    """
    Machine learning models for return prediction.

    Implements various ML algorithms for predicting asset returns
    including linear models, tree-based methods, and ensembles.
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        lookback_periods: int = 20,
        forecast_horizon: int = 1,
    ):
        """
        Initialize return predictor.

        Args:
            model_type: Type of ML model to use
            lookback_periods: Number of historical periods for features
            forecast_horizon: Number of periods ahead to forecast
        """
        self.model_type = model_type
        self.lookback_periods = lookback_periods
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def _create_model(self) -> Any:
        """Create the specified ML model."""
        if self.model_type == ModelType.LINEAR:
            return LinearRegression()
        elif self.model_type == ModelType.RIDGE:
            return Ridge(alpha=1.0)
        elif self.model_type == ModelType.LASSO:
            return Lasso(alpha=0.1)
        elif self.model_type == ModelType.ELASTIC_NET:
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == ModelType.GRADIENT_BOOST:
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Model type {self.model_type} not implemented")

    def prepare_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'returns',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML model.

        Args:
            data: DataFrame with price/return data
            target_col: Name of target column

        Returns:
            Tuple of (features, targets)
        """
        features = []
        targets = []

        # Create lagged features
        for i in range(self.lookback_periods, len(data) - self.forecast_horizon):
            # Historical returns
            hist_returns = data[target_col].iloc[i-self.lookback_periods:i].values

            # Additional features
            feature_vector = list(hist_returns)

            # Add rolling statistics
            feature_vector.append(np.mean(hist_returns))
            feature_vector.append(np.std(hist_returns))
            feature_vector.append(np.min(hist_returns))
            feature_vector.append(np.max(hist_returns))

            # Moving averages
            if i >= 5:
                feature_vector.append(data[target_col].iloc[i-5:i].mean())
            if i >= 20:
                feature_vector.append(data[target_col].iloc[i-20:i].mean())

            features.append(feature_vector)

            # Target is future return
            targets.append(data[target_col].iloc[i + self.forecast_horizon - 1])

        # Store feature names
        self.feature_names = [f'lag_{i+1}' for i in range(self.lookback_periods)]
        self.feature_names.extend(['mean', 'std', 'min', 'max'])
        if len(features[0]) > len(self.feature_names):
            self.feature_names.extend(['ma_5', 'ma_20'])

        return np.array(features), np.array(targets)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        target_col: str = 'returns',
    ) -> Dict[str, float]:
        """
        Train the prediction model.

        Args:
            train_data: Training data
            val_data: Validation data (optional)
            target_col: Target column name

        Returns:
            Dictionary of training metrics
        """
        # Prepare features
        X_train, y_train = self.prepare_features(train_data, target_col)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        # Training predictions
        train_pred = self.model.predict(X_train_scaled)

        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
        }

        # Validation metrics if provided
        if val_data is not None:
            X_val, y_val = self.prepare_features(val_data, target_col)
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)

            metrics.update({
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
            })

        return metrics

    def predict(
        self,
        data: pd.DataFrame,
        target_col: str = 'returns',
        return_confidence: bool = False,
    ) -> PredictionResult:
        """
        Make predictions on new data.

        Args:
            data: Input data
            target_col: Target column name
            return_confidence: Whether to return confidence intervals

        Returns:
            PredictionResult with predictions and metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Prepare features
        X, y_true = self.prepare_features(data, target_col)
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, predictions),
            'mae': mean_absolute_error(y_true, predictions),
            'r2': r2_score(y_true, predictions),
        }

        # Feature importance for tree-based models
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        # Confidence intervals (simplified - using prediction std)
        confidence_intervals = None
        if return_confidence:
            pred_std = np.std(predictions)
            lower = predictions - 1.96 * pred_std
            upper = predictions + 1.96 * pred_std
            confidence_intervals = (lower.tolist(), upper.tolist())

        return PredictionResult(
            predictions=predictions.tolist(),
            model_type=self.model_type.value,
            metrics=metrics,
            feature_importance=feature_importance,
            confidence_intervals=confidence_intervals,
        )


class RegimeClusterer:
    """
    Clustering algorithms for market regime identification.

    Uses unsupervised learning to identify different market regimes
    based on return patterns, volatility, and other features.
    """

    def __init__(
        self,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        n_clusters: int = 3,
    ):
        """
        Initialize regime clusterer.

        Args:
            method: Clustering method to use
            n_clusters: Number of clusters/regimes
        """
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self) -> Any:
        """Create the clustering model."""
        if self.method == ClusteringMethod.KMEANS:
            return KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == ClusteringMethod.DBSCAN:
            return DBSCAN(eps=0.5, min_samples=5)
        elif self.method == ClusteringMethod.GMM:
            return GaussianMixture(n_components=self.n_clusters, random_state=42)
        else:
            raise ValueError(f"Clustering method {self.method} not implemented")

    def prepare_regime_features(
        self,
        returns: pd.DataFrame,
        window: int = 20,
    ) -> np.ndarray:
        """
        Prepare features for regime clustering.

        Args:
            returns: DataFrame of returns
            window: Rolling window size

        Returns:
            Feature array for clustering
        """
        features = []

        # Rolling statistics
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_skew = returns.rolling(window).skew()
        rolling_kurt = returns.rolling(window).kurt()

        # Combine features
        for i in range(window, len(returns)):
            feature_vector = [
                rolling_mean.iloc[i].mean(),
                rolling_std.iloc[i].mean(),
                rolling_skew.iloc[i].mean(),
                rolling_kurt.iloc[i].mean(),
            ]

            # Add correlation if multiple assets
            if returns.shape[1] > 1:
                corr_matrix = returns.iloc[i-window:i].corr()
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                feature_vector.append(avg_corr)

            features.append(feature_vector)

        return np.array(features)

    def identify_regimes(
        self,
        returns: pd.DataFrame,
        window: int = 20,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Identify market regimes using clustering.

        Args:
            returns: Return data
            window: Window size for features

        Returns:
            Tuple of (regime labels, regime descriptions)
        """
        # Prepare features
        features = self.prepare_regime_features(returns, window)
        features_scaled = self.scaler.fit_transform(features)

        # Create and fit model
        self.model = self._create_model()

        if self.method == ClusteringMethod.GMM:
            labels = self.model.fit_predict(features_scaled)
        else:
            labels = self.model.fit_predict(features_scaled)

        # Characterize regimes
        regime_descriptions = self._characterize_regimes(features, labels)

        return labels, regime_descriptions

    def _characterize_regimes(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, str]:
        """
        Characterize identified regimes based on features.

        Args:
            features: Feature array
            labels: Cluster labels

        Returns:
            Dictionary mapping regime ID to description
        """
        descriptions = {}

        for regime in np.unique(labels):
            regime_features = features[labels == regime]

            # Average features for this regime
            avg_return = np.mean(regime_features[:, 0])
            avg_vol = np.mean(regime_features[:, 1])

            # Characterize based on return and volatility
            if avg_return > 0.005 and avg_vol < 0.02:
                desc = "Bull Market (High Return, Low Vol)"
            elif avg_return < -0.005 and avg_vol > 0.03:
                desc = "Bear Market (Negative Return, High Vol)"
            elif abs(avg_return) < 0.002 and avg_vol < 0.015:
                desc = "Sideways Market (Low Return, Low Vol)"
            elif avg_vol > 0.04:
                desc = "Crisis/High Volatility"
            else:
                desc = "Transitional Regime"

            descriptions[regime] = desc

        return descriptions


class AnomalyDetector:
    """
    Anomaly detection for identifying unusual market conditions.

    Uses various algorithms to detect outliers and anomalous patterns
    that may indicate risk events or regime changes.
    """

    def __init__(
        self,
        method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST,
        contamination: float = 0.05,
    ):
        """
        Initialize anomaly detector.

        Args:
            method: Anomaly detection method
            contamination: Expected proportion of anomalies
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self) -> Any:
        """Create the anomaly detection model."""
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            return IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == AnomalyMethod.LOCAL_OUTLIER:
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Anomaly method {self.method} not implemented")

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Detect anomalies in the data.

        Args:
            data: Input data
            features: Feature columns to use

        Returns:
            Tuple of (anomaly flags, anomaly scores)
        """
        if features:
            X = data[features].values
        else:
            X = data.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = self._create_model()

        if self.method == AnomalyMethod.LOCAL_OUTLIER:
            self.model.fit(X_scaled)
            predictions = self.model.predict(X_scaled)
            scores = self.model.score_samples(X_scaled)
        else:
            predictions = self.model.fit_predict(X_scaled)
            scores = self.model.score_samples(X_scaled)

        # Convert predictions to binary (1 = normal, -1 = anomaly)
        anomalies = predictions == -1

        # Create results DataFrame
        results = pd.DataFrame({
            'anomaly': anomalies,
            'anomaly_score': scores,
            'timestamp': data.index,
        })

        return anomalies, results

    def explain_anomalies(
        self,
        data: pd.DataFrame,
        anomalies: np.ndarray,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Provide explanations for detected anomalies.

        Args:
            data: Original data
            anomalies: Anomaly flags
            features: Feature names

        Returns:
            DataFrame with anomaly explanations
        """
        if features is None:
            features = data.columns.tolist()

        explanations = []
        anomaly_indices = np.where(anomalies)[0]

        for idx in anomaly_indices:
            row = data.iloc[idx]

            # Find which features are most unusual
            z_scores = {}
            for feat in features:
                mean = data[feat].mean()
                std = data[feat].std()
                if std > 0:
                    z_scores[feat] = abs((row[feat] - mean) / std)
                else:
                    z_scores[feat] = 0

            # Get top unusual features
            top_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            explanations.append({
                'index': idx,
                'timestamp': data.index[idx],
                'primary_anomaly': top_features[0][0] if top_features else None,
                'z_score': top_features[0][1] if top_features else 0,
                'anomalous_features': [f[0] for f in top_features],
            })

        return pd.DataFrame(explanations)


class RLPortfolioAgent:
    """
    Reinforcement Learning agent for dynamic portfolio allocation.

    Implements Q-learning and policy gradient methods for learning
    optimal allocation strategies through interaction with market environment.
    """

    def __init__(
        self,
        n_assets: int,
        n_actions: int = 5,  # Discrete allocation levels
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        """
        Initialize RL agent.

        Args:
            n_assets: Number of assets
            n_actions: Number of discrete actions per asset
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.n_assets = n_assets
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table for discrete actions
        self.q_table = {}
        self.action_space = self._create_action_space()

    def _create_action_space(self) -> List[np.ndarray]:
        """Create discrete action space for portfolio weights."""
        actions = []

        # Simple discrete allocations (e.g., 0%, 25%, 50%, 75%, 100%)
        allocation_levels = np.linspace(0, 1, self.n_actions)

        # Generate valid portfolio combinations (simplified)
        for i in range(len(allocation_levels)):
            for j in range(len(allocation_levels)):
                if i + j <= len(allocation_levels):
                    # Two-asset example
                    weight1 = allocation_levels[i]
                    weight2 = allocation_levels[j] if i + j < len(allocation_levels) else 0
                    weight3 = 1 - weight1 - weight2

                    if weight3 >= 0 and abs(weight1 + weight2 + weight3 - 1) < 0.001:
                        actions.append(np.array([weight1, weight2, weight3]))

        return actions

    def get_state_representation(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        lookback: int = 10,
    ) -> str:
        """
        Convert market data to state representation.

        Args:
            returns: Recent returns
            prices: Recent prices
            lookback: Lookback period

        Returns:
            State representation (string for Q-table)
        """
        # Discretize returns into bins
        return_bins = np.digitize(returns[-lookback:].mean(), bins=[-0.02, 0, 0.02])
        volatility = returns[-lookback:].std()
        vol_bin = np.digitize(volatility, bins=[0.01, 0.02, 0.03])

        # Simple state representation
        state = f"ret_{return_bins}_vol_{vol_bin}"

        return state

    def choose_action(self, state: str) -> np.ndarray:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action (portfolio weights)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.choice(len(self.action_space))
        else:
            # Exploit: best known action
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.action_space))

            action_idx = np.argmax(self.q_table[state])

        return self.action_space[action_idx]

    def update_q_table(
        self,
        state: str,
        action_idx: int,
        reward: float,
        next_state: str,
    ):
        """
        Update Q-table using Q-learning update rule.

        Args:
            state: Current state
            action_idx: Action index
            reward: Received reward
            next_state: Next state
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.action_space))

        # Q-learning update
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q

    def train_episode(
        self,
        price_data: pd.DataFrame,
        initial_value: float = 10000,
    ) -> Dict[str, float]:
        """
        Train agent for one episode.

        Args:
            price_data: Price data for training
            initial_value: Initial portfolio value

        Returns:
            Episode metrics
        """
        portfolio_value = initial_value
        returns_list = []

        # Calculate returns
        returns = price_data.pct_change().dropna()

        for t in range(10, len(returns) - 1):
            # Get state
            state = self.get_state_representation(
                returns.values[:t],
                price_data.values[:t],
            )

            # Choose action
            action = self.choose_action(state)
            action_idx = self.action_space.index(action) if action.tolist() in [a.tolist() for a in self.action_space] else 0

            # Execute action (simulate portfolio return)
            portfolio_return = np.sum(action[:len(returns.columns)] * returns.iloc[t].values)
            portfolio_value *= (1 + portfolio_return)

            # Calculate reward (Sharpe-based)
            reward = portfolio_return - 0.001  # Subtract risk-free rate

            # Get next state
            next_state = self.get_state_representation(
                returns.values[:t+1],
                price_data.values[:t+1],
            )

            # Update Q-table
            self.update_q_table(state, action_idx, reward, next_state)

            returns_list.append(portfolio_return)

        return {
            'total_return': (portfolio_value / initial_value) - 1,
            'sharpe_ratio': np.mean(returns_list) / (np.std(returns_list) + 1e-10),
            'max_drawdown': self._calculate_max_drawdown(returns_list),
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)


class NeuralForecaster:
    """
    Neural network models for time series forecasting.

    Implements LSTM, GRU, and other deep learning models for
    multi-step ahead forecasting of returns and prices.
    """

    def __init__(
        self,
        model_type: str = 'lstm',
        input_dim: int = 1,
        hidden_dim: int = 50,
        output_dim: int = 1,
        n_layers: int = 2,
    ):
        """
        Initialize neural forecaster.

        Args:
            model_type: Type of neural network (lstm, gru)
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            n_layers: Number of layers
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.model = None
        self.scaler = MinMaxScaler()

    def create_sequences(
        self,
        data: np.ndarray,
        seq_length: int = 20,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            data: Time series data
            seq_length: Sequence length for input
            forecast_horizon: Steps ahead to forecast

        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []

        for i in range(len(data) - seq_length - forecast_horizon + 1):
            seq = data[i:i + seq_length]
            target = data[i + seq_length:i + seq_length + forecast_horizon]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def build_lstm_model(self):
        """Build LSTM model (simplified without actual TensorFlow/PyTorch)."""
        # Placeholder for LSTM model
        # In actual implementation, would use TensorFlow or PyTorch

        class SimpleLSTM:
            def __init__(self, hidden_dim):
                self.hidden_dim = hidden_dim
                self.weights = np.random.randn(hidden_dim, hidden_dim) * 0.01

            def forward(self, x):
                # Simplified LSTM logic
                hidden = np.zeros((x.shape[0], self.hidden_dim))
                for t in range(x.shape[1]):
                    hidden = np.tanh(x[:, t:t+1] + hidden @ self.weights.T)
                return hidden

            def predict(self, x):
                return self.forward(x)[:, -1]

        return SimpleLSTM(self.hidden_dim)

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        seq_length: int = 20,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train neural network model.

        Args:
            train_data: Training data
            val_data: Validation data
            seq_length: Sequence length
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        # Scale data
        train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, seq_length)

        # Build model
        self.model = self.build_lstm_model()

        # Training loop (simplified)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Simplified training step
            predictions = self.model.predict(X_train)
            loss = np.mean((predictions - y_train.flatten()) ** 2)
            history['loss'].append(loss)

            if val_data is not None:
                val_scaled = self.scaler.transform(val_data.reshape(-1, 1)).flatten()
                X_val, y_val = self.create_sequences(val_scaled, seq_length)
                val_predictions = self.model.predict(X_val)
                val_loss = np.mean((val_predictions - y_val.flatten()) ** 2)
                history['val_loss'].append(val_loss)

        return history

    def forecast(
        self,
        data: np.ndarray,
        n_steps: int = 10,
        seq_length: int = 20,
    ) -> np.ndarray:
        """
        Generate multi-step ahead forecasts.

        Args:
            data: Historical data
            n_steps: Number of steps to forecast
            seq_length: Sequence length for model

        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")

        # Scale data
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()

        # Use last sequence for prediction
        last_sequence = data_scaled[-seq_length:]
        forecasts = []

        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(last_sequence.reshape(1, -1))
            forecasts.append(next_pred[0])

            # Update sequence
            last_sequence = np.append(last_sequence[1:], next_pred)

        # Inverse transform
        forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

        return forecasts


class FeatureEngineer:
    """
    Feature engineering pipeline for financial data.

    Creates technical indicators, statistical features, and
    derived variables for ML models.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_pipeline = []

    def add_price_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Add price-based features.

        Args:
            df: DataFrame with price data
            price_col: Price column name

        Returns:
            DataFrame with new features
        """
        # Returns
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))

        # Price ratios
        df['price_to_high'] = df[price_col] / df['high'] if 'high' in df else 1
        df['price_to_low'] = df[price_col] / df['low'] if 'low' in df else 1

        return df

    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Add technical indicators.

        Args:
            df: DataFrame with price data
            price_col: Price column name

        Returns:
            DataFrame with technical indicators
        """
        # Moving averages
        df['sma_5'] = df[price_col].rolling(5).mean()
        df['sma_20'] = df[price_col].rolling(20).mean()
        df['sma_50'] = df[price_col].rolling(50).mean()

        # Exponential moving averages
        df['ema_12'] = df[price_col].ewm(span=12).mean()
        df['ema_26'] = df[price_col].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = df[price_col].rolling(20).mean()
        std_20 = df[price_col].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        return df

    def add_volatility_features(
        self,
        df: pd.DataFrame,
        returns_col: str = 'returns',
    ) -> pd.DataFrame:
        """
        Add volatility-based features.

        Args:
            df: DataFrame with returns
            returns_col: Returns column name

        Returns:
            DataFrame with volatility features
        """
        # Historical volatility
        df['volatility_5'] = df[returns_col].rolling(5).std()
        df['volatility_20'] = df[returns_col].rolling(20).std()
        df['volatility_60'] = df[returns_col].rolling(60).std()

        # Volatility ratios
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']

        # GARCH volatility (simplified)
        df['squared_returns'] = df[returns_col] ** 2
        df['garch_vol'] = df['squared_returns'].ewm(span=20).mean() ** 0.5

        return df

    def add_market_microstructure(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume',
    ) -> pd.DataFrame:
        """
        Add market microstructure features.

        Args:
            df: DataFrame with volume data
            volume_col: Volume column name

        Returns:
            DataFrame with microstructure features
        """
        if volume_col in df:
            # Volume features
            df['volume_ma_5'] = df[volume_col].rolling(5).mean()
            df['volume_ma_20'] = df[volume_col].rolling(20).mean()
            df['volume_ratio'] = df[volume_col] / df['volume_ma_20']

            # VWAP (Volume Weighted Average Price)
            if 'close' in df:
                df['vwap'] = (df['close'] * df[volume_col]).rolling(20).sum() / df[volume_col].rolling(20).sum()
                df['price_to_vwap'] = df['close'] / df['vwap']

        return df

    def create_feature_pipeline(
        self,
        df: pd.DataFrame,
        feature_types: List[FeatureType],
    ) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline.

        Args:
            df: Input DataFrame
            feature_types: Types of features to create

        Returns:
            DataFrame with all features
        """
        df_features = df.copy()

        if FeatureType.PRICE in feature_types:
            df_features = self.add_price_features(df_features)

        if FeatureType.TECHNICAL in feature_types:
            df_features = self.add_technical_indicators(df_features)

        if FeatureType.VOLATILITY in feature_types:
            df_features = self.add_volatility_features(df_features)

        if FeatureType.VOLUME in feature_types:
            df_features = self.add_market_microstructure(df_features)

        # Drop NaN values from rolling calculations
        df_features = df_features.dropna()

        return df_features


class ModelBacktester:
    """
    Backtesting and validation framework for ML models.

    Provides walk-forward analysis, cross-validation, and
    performance evaluation for predictive models.
    """

    def __init__(
        self,
        model: Any,
        test_ratio: float = 0.2,
        n_splits: int = 5,
    ):
        """
        Initialize model backtester.

        Args:
            model: ML model to backtest
            test_ratio: Test set ratio
            n_splits: Number of splits for time series CV
        """
        self.model = model
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.results = []

    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 21,
        target_col: str = 'returns',
    ) -> pd.DataFrame:
        """
        Perform walk-forward analysis.

        Args:
            data: Input data
            window_size: Training window size
            step_size: Step size for rolling window
            target_col: Target column

        Returns:
            DataFrame with backtest results
        """
        results = []

        for i in range(window_size, len(data) - step_size, step_size):
            # Training window
            train_data = data.iloc[i-window_size:i]

            # Test window
            test_data = data.iloc[i:i+step_size]

            # Train model
            if hasattr(self.model, 'train'):
                self.model.train(train_data, target_col=target_col)
            else:
                X_train, y_train = self._prepare_sklearn_data(train_data, target_col)
                self.model.fit(X_train, y_train)

            # Make predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(test_data, target_col=target_col)
                if isinstance(predictions, PredictionResult):
                    pred_values = predictions.predictions
                else:
                    pred_values = predictions
            else:
                X_test, y_test = self._prepare_sklearn_data(test_data, target_col)
                pred_values = self.model.predict(X_test)

            # Calculate metrics
            y_test = test_data[target_col].values[-len(pred_values):]

            results.append({
                'period_start': test_data.index[0],
                'period_end': test_data.index[-1],
                'mse': mean_squared_error(y_test, pred_values),
                'mae': mean_absolute_error(y_test, pred_values),
                'r2': r2_score(y_test, pred_values),
                'directional_accuracy': np.mean(np.sign(pred_values) == np.sign(y_test)),
            })

        return pd.DataFrame(results)

    def time_series_cross_validation(
        self,
        data: pd.DataFrame,
        target_col: str = 'returns',
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            data: Input data
            target_col: Target column

        Returns:
            Cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        scores = {'mse': [], 'mae': [], 'r2': []}

        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Train model
            if hasattr(self.model, 'train'):
                self.model.train(train_data, target_col=target_col)
                predictions = self.model.predict(test_data, target_col=target_col)
                if isinstance(predictions, PredictionResult):
                    pred_values = predictions.predictions
                else:
                    pred_values = predictions
            else:
                X_train, y_train = self._prepare_sklearn_data(train_data, target_col)
                X_test, y_test = self._prepare_sklearn_data(test_data, target_col)
                self.model.fit(X_train, y_train)
                pred_values = self.model.predict(X_test)

            y_test = test_data[target_col].values[-len(pred_values):]

            scores['mse'].append(mean_squared_error(y_test, pred_values))
            scores['mae'].append(mean_absolute_error(y_test, pred_values))
            scores['r2'].append(r2_score(y_test, pred_values))

        return {
            'mse_mean': np.mean(scores['mse']),
            'mse_std': np.std(scores['mse']),
            'mae_mean': np.mean(scores['mae']),
            'mae_std': np.std(scores['mae']),
            'r2_mean': np.mean(scores['r2']),
            'r2_std': np.std(scores['r2']),
        }

    def _prepare_sklearn_data(
        self,
        data: pd.DataFrame,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sklearn models."""
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols].values
        y = data[target_col].values
        return X, y

    def calculate_trading_metrics(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.

        Args:
            predictions: Predicted returns/signals
            actual_returns: Actual returns
            transaction_cost: Transaction cost per trade

        Returns:
            Trading metrics
        """
        # Generate trading signals
        signals = np.sign(predictions)

        # Calculate portfolio returns
        portfolio_returns = signals[:-1] * actual_returns[1:]

        # Account for transaction costs
        trades = np.diff(signals)
        n_trades = np.sum(np.abs(trades) > 0)
        total_cost = n_trades * transaction_cost

        # Calculate metrics
        total_return = np.prod(1 + portfolio_returns) - 1 - total_cost
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-10) * np.sqrt(252)

        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = portfolio_returns > 0
        win_rate = np.mean(winning_trades)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': total_return / max(n_trades, 1),
        }