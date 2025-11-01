"""
Machine Learning Integration Examples

This script demonstrates the machine learning capabilities of Allocation Station:
1. Return prediction models (Random Forest, Gradient Boosting, etc.)
2. Clustering for regime identification (K-means, GMM, DBSCAN)
3. Anomaly detection for risk events (Isolation Forest, LOF)
4. Reinforcement learning for dynamic allocation (Q-learning)
5. Neural network forecasting (LSTM, GRU)
6. Feature engineering pipeline
7. Model backtesting and validation

Run this script to see each ML capability in action.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from allocation_station.ml.ml_models import (
    ReturnPredictor,
    ModelType,
    RegimeClusterer,
    ClusteringMethod,
    AnomalyDetector,
    AnomalyMethod,
    RLPortfolioAgent,
    NeuralForecaster,
    FeatureEngineer,
    FeatureType,
    ModelBacktester,
)


def generate_sample_market_data(n_days=1000, n_assets=5):
    """Generate sample market data for ML examples."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Generate correlated asset prices
    mean_returns = np.array([0.08, 0.06, 0.10, 0.04, 0.05]) / 252  # Annual to daily
    volatilities = np.array([0.20, 0.15, 0.25, 0.10, 0.18]) / np.sqrt(252)

    # Correlation matrix
    correlation = np.array([
        [1.00, 0.60, 0.50, 0.20, 0.40],
        [0.60, 1.00, 0.45, 0.25, 0.35],
        [0.50, 0.45, 1.00, 0.15, 0.30],
        [0.20, 0.25, 0.15, 1.00, 0.10],
        [0.40, 0.35, 0.30, 0.10, 1.00],
    ])

    cov_matrix = np.outer(volatilities, volatilities) * correlation

    # Generate returns with some regime structure
    returns = []
    regime = 0  # Start in normal regime

    for i in range(n_days):
        # Occasionally switch regime
        if np.random.random() < 0.02:  # 2% chance to switch
            regime = (regime + 1) % 3

        if regime == 0:  # Normal
            day_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
        elif regime == 1:  # Bull
            day_returns = np.random.multivariate_normal(mean_returns * 2, cov_matrix * 0.8)
        else:  # Bear
            day_returns = np.random.multivariate_normal(mean_returns * -1.5, cov_matrix * 1.5)

        returns.append(day_returns)

    returns = np.array(returns)

    # Calculate prices
    prices = np.zeros((n_days, n_assets))
    prices[0] = 100  # Starting price

    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])

    # Create DataFrames
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

    price_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

    # Add volume data
    volumes = np.random.lognormal(14, 1.5, (n_days, n_assets))  # Log-normal volumes
    volume_df = pd.DataFrame(volumes, index=dates, columns=asset_names)

    return price_df, returns_df, volume_df


def example_1_return_prediction():
    """Example 1: Return prediction using ML models."""
    print("=" * 80)
    print("EXAMPLE 1: Return Prediction Models")
    print("=" * 80)

    # Generate sample data
    price_df, returns_df, _ = generate_sample_market_data(n_days=500)

    # Focus on single asset for simplicity
    asset_returns = returns_df['Asset_1'].to_frame('returns')

    print("\nData Summary:")
    print(f"  Total days: {len(asset_returns)}")
    print(f"  Mean return: {asset_returns['returns'].mean():.4%}")
    print(f"  Volatility: {asset_returns['returns'].std():.4%}")

    # Split data
    train_size = int(0.7 * len(asset_returns))
    train_data = asset_returns[:train_size]
    test_data = asset_returns[train_size:]

    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_data)} days")
    print(f"  Testing: {len(test_data)} days")

    # Test different models
    models = [
        (ModelType.LINEAR, "Linear Regression"),
        (ModelType.RIDGE, "Ridge Regression"),
        (ModelType.RANDOM_FOREST, "Random Forest"),
        (ModelType.GRADIENT_BOOST, "Gradient Boosting"),
    ]

    print("\n--- Model Performance Comparison ---")

    results = []
    for model_type, model_name in models:
        print(f"\n{model_name}:")

        # Create and train predictor
        predictor = ReturnPredictor(
            model_type=model_type,
            lookback_periods=10,
            forecast_horizon=1,
        )

        # Train model
        train_metrics = predictor.train(train_data, target_col='returns')

        print(f"  Training R²: {train_metrics['train_r2']:.3f}")
        print(f"  Training MSE: {train_metrics['train_mse']:.6f}")

        # Test predictions
        test_results = predictor.predict(test_data, target_col='returns')

        print(f"  Test R²: {test_results.metrics['r2']:.3f}")
        print(f"  Test MSE: {test_results.metrics['mse']:.6f}")
        print(f"  Test MAE: {test_results.metrics['mae']:.6f}")

        # Directional accuracy
        actual = test_data['returns'].values[predictor.lookback_periods:predictor.lookback_periods+len(test_results.predictions)]
        predicted = np.array(test_results.predictions)
        directional_accuracy = np.mean(np.sign(actual) == np.sign(predicted))
        print(f"  Directional Accuracy: {directional_accuracy:.1%}")

        results.append({
            'model': model_name,
            'test_r2': test_results.metrics['r2'],
            'directional_acc': directional_accuracy,
        })

        # Show feature importance for tree-based models
        if test_results.feature_importance:
            print(f"\n  Top 3 Important Features:")
            sorted_features = sorted(test_results.feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)[:3]
            for feat, importance in sorted_features:
                print(f"    {feat}: {importance:.3f}")

    # Summary
    print("\n--- Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


def example_2_regime_clustering():
    """Example 2: Market regime identification using clustering."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Regime Identification with Clustering")
    print("=" * 80)

    # Generate data with regime changes
    price_df, returns_df, _ = generate_sample_market_data(n_days=750)

    print(f"\nAnalyzing {len(returns_df)} days of returns for {len(returns_df.columns)} assets")

    # Test different clustering methods
    methods = [
        (ClusteringMethod.KMEANS, 3, "K-Means"),
        (ClusteringMethod.GMM, 3, "Gaussian Mixture Model"),
    ]

    for method, n_clusters, method_name in methods:
        print(f"\n--- {method_name} Clustering ---")

        # Create clusterer
        clusterer = RegimeClusterer(method=method, n_clusters=n_clusters)

        # Identify regimes
        regimes, descriptions = clusterer.identify_regimes(returns_df, window=20)

        print(f"\nIdentified {len(descriptions)} regimes:")
        for regime_id, description in descriptions.items():
            regime_count = np.sum(regimes == regime_id)
            regime_pct = regime_count / len(regimes) * 100
            print(f"  Regime {regime_id}: {description}")
            print(f"           Count: {regime_count} periods ({regime_pct:.1f}%)")

        # Analyze returns by regime
        print("\nReturns by Regime:")
        for regime_id in descriptions.keys():
            regime_mask = regimes == regime_id
            if np.any(regime_mask):
                # Need to align indices properly
                valid_indices = np.where(regime_mask)[0]
                regime_returns = []

                for idx in valid_indices:
                    actual_idx = idx + 20  # Offset by window size
                    if actual_idx < len(returns_df):
                        regime_returns.append(returns_df.iloc[actual_idx].mean())

                if regime_returns:
                    avg_return = np.mean(regime_returns)
                    avg_vol = np.std(regime_returns)
                    print(f"  Regime {regime_id}: Return={avg_return:.4%}, Vol={avg_vol:.4%}")

        # Show regime transitions
        transitions = np.diff(regimes)
        n_transitions = np.sum(transitions != 0)
        print(f"\nRegime Transitions: {n_transitions} ({n_transitions/len(regimes)*100:.1f}% of periods)")


def example_3_anomaly_detection():
    """Example 3: Anomaly detection for risk events."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Anomaly Detection for Risk Events")
    print("=" * 80)

    # Generate data with some anomalies
    price_df, returns_df, volume_df = generate_sample_market_data(n_days=500)

    # Inject some anomalies
    anomaly_dates = [50, 150, 250, 350, 450]
    for date_idx in anomaly_dates:
        # Large negative return
        returns_df.iloc[date_idx] = returns_df.iloc[date_idx] * np.random.uniform(-3, -5)
        # Spike in volume
        volume_df.iloc[date_idx] = volume_df.iloc[date_idx] * np.random.uniform(3, 5)

    print(f"\nData with {len(anomaly_dates)} injected anomalies")

    # Create features for anomaly detection
    features_df = pd.DataFrame({
        'returns_mean': returns_df.mean(axis=1),
        'returns_std': returns_df.std(axis=1),
        'volume_mean': volume_df.mean(axis=1),
        'volume_std': volume_df.std(axis=1),
    })

    # Test different anomaly detection methods
    methods = [
        (AnomalyMethod.ISOLATION_FOREST, "Isolation Forest"),
        (AnomalyMethod.LOCAL_OUTLIER, "Local Outlier Factor"),
    ]

    for method, method_name in methods:
        print(f"\n--- {method_name} ---")

        # Create detector
        detector = AnomalyDetector(
            method=method,
            contamination=0.02,  # Expect 2% anomalies
        )

        # Detect anomalies
        anomalies, results_df = detector.detect_anomalies(features_df)

        n_detected = np.sum(anomalies)
        print(f"  Detected {n_detected} anomalies ({n_detected/len(anomalies)*100:.1f}%)")

        # Check detection accuracy for injected anomalies
        true_positives = 0
        for date_idx in anomaly_dates:
            if date_idx < len(anomalies) and anomalies[date_idx]:
                true_positives += 1

        print(f"  True Positives: {true_positives}/{len(anomaly_dates)} injected anomalies")

        # Get explanations for detected anomalies
        explanations = detector.explain_anomalies(features_df, anomalies)

        if len(explanations) > 0:
            print(f"\n  Top 3 Anomalies:")
            for _, row in explanations.head(3).iterrows():
                print(f"    Date index {row['index']}: {row['primary_anomaly']} (z={row['z_score']:.2f})")

        # Show anomaly scores distribution
        print(f"\n  Anomaly Score Statistics:")
        print(f"    Min:    {results_df['anomaly_score'].min():.3f}")
        print(f"    Mean:   {results_df['anomaly_score'].mean():.3f}")
        print(f"    Max:    {results_df['anomaly_score'].max():.3f}")
        print(f"    Std:    {results_df['anomaly_score'].std():.3f}")


def example_4_reinforcement_learning():
    """Example 4: Reinforcement learning for portfolio allocation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Reinforcement Learning Portfolio Agent")
    print("=" * 80)

    # Generate sample data
    price_df, returns_df, _ = generate_sample_market_data(n_days=500, n_assets=3)

    print(f"\nTraining RL agent on {len(price_df)} days, {len(price_df.columns)} assets")

    # Create RL agent
    agent = RLPortfolioAgent(
        n_assets=3,
        n_actions=5,  # 5 discrete allocation levels
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.2,  # 20% exploration
    )

    print("\nRL Agent Configuration:")
    print(f"  Assets: {agent.n_assets}")
    print(f"  Action Space Size: {len(agent.action_space)}")
    print(f"  Learning Rate: {agent.learning_rate}")
    print(f"  Discount Factor: {agent.gamma}")
    print(f"  Exploration Rate: {agent.epsilon}")

    # Training episodes
    n_episodes = 10
    episode_results = []

    print(f"\n--- Training for {n_episodes} Episodes ---")

    for episode in range(n_episodes):
        # Train for one episode
        metrics = agent.train_episode(price_df)
        episode_results.append(metrics)

        if (episode + 1) % 2 == 0:
            print(f"  Episode {episode+1}:")
            print(f"    Total Return: {metrics['total_return']:.2%}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Show learning progress
    print("\n--- Learning Progress ---")
    returns = [r['total_return'] for r in episode_results]
    sharpes = [r['sharpe_ratio'] for r in episode_results]

    print(f"First 3 episodes avg return: {np.mean(returns[:3]):.2%}")
    print(f"Last 3 episodes avg return:  {np.mean(returns[-3:]):.2%}")
    print(f"Return improvement: {(np.mean(returns[-3:]) - np.mean(returns[:3])):.2%}")

    print(f"\nFirst 3 episodes avg Sharpe: {np.mean(sharpes[:3]):.3f}")
    print(f"Last 3 episodes avg Sharpe:  {np.mean(sharpes[-3:]):.3f}")

    # Show Q-table statistics
    print(f"\n--- Q-Table Statistics ---")
    print(f"States explored: {len(agent.q_table)}")

    if agent.q_table:
        all_q_values = []
        for state_q in agent.q_table.values():
            all_q_values.extend(state_q)

        print(f"Q-value range: [{min(all_q_values):.3f}, {max(all_q_values):.3f}]")
        print(f"Q-value mean: {np.mean(all_q_values):.3f}")

    # Reduce exploration for final test
    agent.epsilon = 0.05
    print(f"\n--- Final Test (epsilon={agent.epsilon}) ---")

    test_metrics = agent.train_episode(price_df[-200:])  # Test on last 200 days
    print(f"Test Return: {test_metrics['total_return']:.2%}")
    print(f"Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
    print(f"Test Max DD: {test_metrics['max_drawdown']:.2%}")


def example_5_neural_forecasting():
    """Example 5: Neural network time series forecasting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Neural Network Forecasting")
    print("=" * 80)

    # Generate sample data
    price_df, returns_df, _ = generate_sample_market_data(n_days=500)

    # Use single asset for simplicity
    asset_prices = price_df['Asset_1'].values

    print(f"\nTime Series Data:")
    print(f"  Length: {len(asset_prices)} days")
    print(f"  Mean: ${np.mean(asset_prices):.2f}")
    print(f"  Std: ${np.std(asset_prices):.2f}")

    # Create neural forecaster
    forecaster = NeuralForecaster(
        model_type='lstm',
        input_dim=1,
        hidden_dim=20,
        output_dim=1,
        n_layers=2,
    )

    print("\nNeural Network Configuration:")
    print(f"  Model Type: {forecaster.model_type.upper()}")
    print(f"  Hidden Dimension: {forecaster.hidden_dim}")
    print(f"  Number of Layers: {forecaster.n_layers}")

    # Split data
    train_size = int(0.8 * len(asset_prices))
    train_data = asset_prices[:train_size]
    test_data = asset_prices[train_size:]

    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_data)} days")
    print(f"  Testing: {len(test_data)} days")

    # Train model
    print("\n--- Training Neural Network ---")
    history = forecaster.train(
        train_data,
        val_data=test_data,
        seq_length=20,
        epochs=10,
        batch_size=32,
    )

    print(f"  Final Training Loss: {history['loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"  Final Validation Loss: {history['val_loss'][-1]:.6f}")

    # Make forecasts
    print("\n--- Multi-Step Forecasting ---")
    n_forecast_steps = 10

    forecasts = forecaster.forecast(
        train_data,
        n_steps=n_forecast_steps,
        seq_length=20,
    )

    print(f"\n{n_forecast_steps}-Day Forecast:")
    for i, forecast in enumerate(forecasts[:5], 1):
        print(f"  Day {i}: ${forecast:.2f}")

    # Compare with actual if available
    if len(test_data) >= n_forecast_steps:
        actual = test_data[:n_forecast_steps]
        forecast_error = np.mean(np.abs(forecasts - actual))
        print(f"\nForecast MAE: ${forecast_error:.2f}")

        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        forecast_direction = np.diff(forecasts) > 0
        directional_acc = np.mean(actual_direction == forecast_direction)
        print(f"Directional Accuracy: {directional_acc:.1%}")


def example_6_feature_engineering():
    """Example 6: Feature engineering pipeline."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Feature Engineering Pipeline")
    print("=" * 80)

    # Generate sample data with OHLCV
    price_df, returns_df, volume_df = generate_sample_market_data(n_days=200)

    # Create OHLCV data for single asset
    asset = 'Asset_1'
    ohlcv_df = pd.DataFrame({
        'open': price_df[asset] * np.random.uniform(0.98, 1.02, len(price_df)),
        'high': price_df[asset] * np.random.uniform(1.01, 1.03, len(price_df)),
        'low': price_df[asset] * np.random.uniform(0.97, 0.99, len(price_df)),
        'close': price_df[asset],
        'volume': volume_df[asset],
    })

    print(f"Input Data Shape: {ohlcv_df.shape}")
    print(f"Columns: {list(ohlcv_df.columns)}")

    # Create feature engineer
    engineer = FeatureEngineer()

    # Apply feature engineering pipeline
    print("\n--- Applying Feature Engineering Pipeline ---")

    feature_types = [
        FeatureType.PRICE,
        FeatureType.TECHNICAL,
        FeatureType.VOLATILITY,
        FeatureType.VOLUME,
    ]

    features_df = engineer.create_feature_pipeline(ohlcv_df, feature_types)

    print(f"\nOutput Features Shape: {features_df.shape}")
    print(f"Number of Features: {len(features_df.columns)}")

    # Show feature categories
    print("\n--- Feature Categories ---")

    # Price features
    price_features = [col for col in features_df.columns if 'price' in col.lower() or 'return' in col.lower()]
    print(f"\nPrice/Return Features ({len(price_features)}):")
    for feat in price_features[:5]:
        print(f"  - {feat}")

    # Technical indicators
    tech_features = [col for col in features_df.columns if any(ind in col.lower() for ind in ['sma', 'ema', 'macd', 'rsi', 'bb'])]
    print(f"\nTechnical Indicators ({len(tech_features)}):")
    for feat in tech_features[:5]:
        print(f"  - {feat}")

    # Volatility features
    vol_features = [col for col in features_df.columns if 'vol' in col.lower()]
    print(f"\nVolatility Features ({len(vol_features)}):")
    for feat in vol_features[:5]:
        print(f"  - {feat}")

    # Show feature statistics
    print("\n--- Feature Statistics ---")
    feature_stats = features_df.describe().T[['mean', 'std', 'min', 'max']]
    print(feature_stats.head(10))

    # Check for highly correlated features
    print("\n--- Feature Correlation Analysis ---")
    corr_matrix = features_df.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.95)")
        for feat1, feat2, corr in high_corr_pairs[:3]:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("No highly correlated features found (|r| > 0.95)")


def example_7_model_backtesting():
    """Example 7: Model backtesting and validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Model Backtesting and Validation")
    print("=" * 80)

    # Generate sample data
    price_df, returns_df, _ = generate_sample_market_data(n_days=750)

    # Prepare data with features
    data_df = pd.DataFrame({
        'returns': returns_df['Asset_1'],
        'lag_1': returns_df['Asset_1'].shift(1),
        'lag_2': returns_df['Asset_1'].shift(2),
        'lag_3': returns_df['Asset_1'].shift(3),
        'ma_5': returns_df['Asset_1'].rolling(5).mean(),
        'vol_20': returns_df['Asset_1'].rolling(20).std(),
    }).dropna()

    print(f"Backtesting Dataset: {len(data_df)} observations")

    # Create predictor model
    predictor = ReturnPredictor(
        model_type=ModelType.RANDOM_FOREST,
        lookback_periods=5,
        forecast_horizon=1,
    )

    # Create backtester
    backtester = ModelBacktester(
        model=predictor,
        test_ratio=0.2,
        n_splits=5,
    )

    # 1. Walk-Forward Analysis
    print("\n--- Walk-Forward Analysis ---")
    print("  Window Size: 252 days (1 year)")
    print("  Step Size: 21 days (1 month)")

    wf_results = backtester.walk_forward_analysis(
        data_df,
        window_size=252,
        step_size=21,
        target_col='returns',
    )

    print(f"\nWalk-Forward Results ({len(wf_results)} periods):")
    print(f"  Average MSE: {wf_results['mse'].mean():.6f}")
    print(f"  Average MAE: {wf_results['mae'].mean():.6f}")
    print(f"  Average R²: {wf_results['r2'].mean():.3f}")
    print(f"  Avg Directional Accuracy: {wf_results['directional_accuracy'].mean():.1%}")

    # Show performance over time
    print("\nPerformance Stability:")
    print(f"  R² Std Dev: {wf_results['r2'].std():.3f}")
    print(f"  Best R²: {wf_results['r2'].max():.3f} (Period {wf_results['r2'].idxmax() + 1})")
    print(f"  Worst R²: {wf_results['r2'].min():.3f} (Period {wf_results['r2'].idxmin() + 1})")

    # 2. Time Series Cross-Validation
    print("\n--- Time Series Cross-Validation ---")

    cv_results = backtester.time_series_cross_validation(
        data_df,
        target_col='returns',
    )

    print(f"\nCross-Validation Results ({backtester.n_splits} splits):")
    print(f"  MSE: {cv_results['mse_mean']:.6f} ± {cv_results['mse_std']:.6f}")
    print(f"  MAE: {cv_results['mae_mean']:.6f} ± {cv_results['mae_std']:.6f}")
    print(f"  R²:  {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")

    # 3. Trading Strategy Metrics
    print("\n--- Trading Strategy Performance ---")

    # Generate predictions for trading simulation
    train_size = int(0.7 * len(data_df))
    train_data = data_df[:train_size]
    test_data = data_df[train_size:]

    # Train model
    predictor.train(train_data, target_col='returns')

    # Get predictions
    pred_result = predictor.predict(test_data, target_col='returns')
    predictions = np.array(pred_result.predictions)

    # Calculate trading metrics - ensure arrays are aligned
    # predictions length determines the actual_returns slice
    min_length = min(len(predictions), len(test_data) - predictor.lookback_periods)
    predictions = predictions[:min_length]
    actual_returns = test_data['returns'].values[predictor.lookback_periods:predictor.lookback_periods+min_length]

    trading_metrics = backtester.calculate_trading_metrics(
        predictions,
        actual_returns,
        transaction_cost=0.001,
    )

    print("\nTrading Strategy Results:")
    print(f"  Total Return: {trading_metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {trading_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {trading_metrics['n_trades']:.0f}")
    print(f"  Win Rate: {trading_metrics['win_rate']:.1%}")
    print(f"  Avg Return per Trade: {trading_metrics['avg_return_per_trade']:.3%}")

    # Compare with buy-and-hold
    buy_hold_return = np.prod(1 + actual_returns) - 1
    print(f"\nBuy-and-Hold Return: {buy_hold_return:.2%}")
    print(f"Strategy Outperformance: {(trading_metrics['total_return'] - buy_hold_return):.2%}")


def main():
    """Run all examples."""
    print("\n")
    print("#" * 80)
    print("# MACHINE LEARNING INTEGRATION EXAMPLES")
    print("#" * 80)
    print("\nThis script demonstrates ML capabilities")
    print("implemented in Allocation Station.\n")

    try:
        # Note: Some ML examples require additional implementation
        # The ML models module is a placeholder and needs full implementation
        print("\nNote: ML examples are placeholders pending full implementation.")
        print("The ML integration module provides interfaces for:")
        print("  - Return prediction models (Linear, Ridge, RF, GBM)")
        print("  - Clustering for regime identification")
        print("  - Anomaly detection for risk events")
        print("  - Reinforcement learning for dynamic allocation")
        print("  - Neural network forecasting")
        print("  - Feature engineering pipeline")
        print("  - Model backtesting and validation")

        # Uncomment when ML models are fully implemented:
        # example_1_return_prediction()
        # example_2_regime_clustering()
        # example_3_anomaly_detection()
        # example_4_reinforcement_learning()
        # example_5_neural_forecasting()
        # example_6_feature_engineering()
        # example_7_model_backtesting()

        print("\n" + "=" * 80)
        print("ML Integration examples listed (full implementation pending)")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()