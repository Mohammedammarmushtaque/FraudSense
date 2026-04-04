import joblib
import numpy as np
from datetime import datetime
import os

# Internal mock classes for when real models are missing
class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))
    def predict_proba(self, X):
        # Return [prob_legit, prob_fraud]
        return np.zeros((len(X), 2))
    def decision_function(self, X):
        return np.zeros(len(X))

class DummyScaler:
    def transform(self, X):
        return X

# Load models once at module level
MODEL_DIR = 'models'

try:
    if os.path.exists(os.path.join(MODEL_DIR, 'isolation_forest.pkl')):
        isolation_forest = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
        xgboost_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
        feature_scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
        print("ML models loaded successfully")
    else:
        raise FileNotFoundError("Models directory or files missing")
except Exception as e:
    print(f"Warning: Could not load models - {e}")
    print("Run train_models.py first to generate model files")
    isolation_forest = DummyModel()
    xgboost_model = DummyModel()
    feature_scaler = DummyScaler()

# Transaction type encoding mapping
TX_TYPE_ENCODING = {
    'PAYMENT': 0,
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
}


def score_transaction(tx_data, user_profile):
    """
    Score a transaction using ML ensemble of Isolation Forest and XGBoost
    
    Args:
        tx_data: dict with keys {amount, merchant_id, timestamp, oldbalanceOrg, 
                 newbalanceOrig, oldbalanceDest, newbalanceDest, tx_type}
        user_profile: dict from behavioral.get_user_profile
        
    Returns:
        dict with {ml_score, if_score, xgb_score} (0-100 scale)
    """
    
    # Check if models are loaded
    if isolation_forest is None or xgboost_model is None or feature_scaler is None:
        print("⚠ Models not loaded, returning default scores")
        return {
            "ml_score": 50.0,  # Neutral score
            "if_score": 50.0,
            "xgb_score": 50.0
        }
    
    try:
        # Extract transaction features with safe defaults
        amount = float(tx_data.get('amount', 0))
        oldbalance_orig = float(tx_data.get('oldbalanceOrg', 0))
        newbalance_orig = float(tx_data.get('newbalanceOrig', 0))
        oldbalance_dest = float(tx_data.get('oldbalanceDest', 0))
        newbalance_dest = float(tx_data.get('newbalanceDest', 0))
        tx_type = tx_data.get('tx_type', 'PAYMENT')
        timestamp = tx_data.get('timestamp', datetime.now())
        
        # Feature 1: amount
        feature_amount = amount
        
        # Feature 2: balance_diff_orig
        balance_diff_orig = oldbalance_orig - newbalance_orig
        
        # Feature 3: balance_diff_dest
        balance_diff_dest = newbalance_dest - oldbalance_dest
        
        # Feature 4: hour_of_day
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        hour_of_day = timestamp.hour
        
        # Feature 5: tx_type_encoded
        tx_type_encoded = TX_TYPE_ENCODING.get(tx_type.upper(), 0)
        
        # Feature 6: amount_to_balance_ratio
        amount_to_balance_ratio = amount / (oldbalance_orig + 1)  # Add 1 to avoid division by zero
        
        # Create feature vector (same order as training)
        feature_vector = np.array([[
            feature_amount,
            balance_diff_orig,
            balance_diff_dest,
            hour_of_day,
            tx_type_encoded,
            amount_to_balance_ratio
        ]])
        
        # Scale features
        scaled_features = feature_scaler.transform(feature_vector)
        
        # ISOLATION FOREST SCORE
        # decision_function returns: positive for inliers, negative for outliers
        raw_if_score = isolation_forest.decision_function(scaled_features)[0]
        
        # Normalize to 0-100 scale (inliers → 0, outliers → 100)
        # raw_if_score typically ranges from -0.5 to 0.5
        # Clamp and normalize
        normalized_if = (1 - raw_if_score) / 2  # Maps [-0.5, 0.5] to [0.75, 0.25]
        if_score = max(0, min(100, normalized_if * 100))
        
        # XGBOOST SCORE
        # predict_proba returns [prob_legitimate, prob_fraud]
        fraud_prob = xgboost_model.predict_proba(scaled_features)[0][1]
        xgb_score = fraud_prob * 100
        
        # ML ENSEMBLE SCORE (weighted combination)
        # Isolation Forest: 30%, XGBoost: 70%
        ml_score = if_score * 0.3 + xgb_score * 0.7
        
        return {
            "ml_score": round(ml_score, 2),
            "if_score": round(if_score, 2),
            "xgb_score": round(xgb_score, 2)
        }
        
    except Exception as e:
        print(f"⚠ Error scoring transaction: {e}")
        # Return neutral scores on error
        return {
            "ml_score": 50.0,
            "if_score": 50.0,
            "xgb_score": 50.0
        }


def get_feature_importance():
    """
    Return feature importance from XGBoost model
    
    Returns:
        dict mapping feature names to importance scores
    """
    if xgboost_model is None:
        return {}
    
    feature_names = [
        'amount',
        'balance_diff_orig',
        'balance_diff_dest',
        'hour_of_day',
        'tx_type_encoded',
        'amount_to_balance_ratio'
    ]
    
    try:
        # Get feature importance from XGBoost
        importance = xgboost_model.feature_importances_
        
        feature_importance = {}
        for name, imp in zip(feature_names, importance):
            feature_importance[name] = round(float(imp), 4)
        
        return feature_importance
    except Exception as e:
        print(f"⚠ Error getting feature importance: {e}")
        return {}


def interpret_score(ml_score):
    """
    Interpret ML score into risk category
    
    Args:
        ml_score: float (0-100)
        
    Returns:
        str: risk category
    """
    if ml_score < 30:
        return "LOW"
    elif ml_score < 60:
        return "MEDIUM"
    elif ml_score < 80:
        return "HIGH"
    else:
        return "CRITICAL"


def calculate_final_risk_score(component_scores: dict, weights: dict) -> float:
    """
    Standardizes and combines component scores into a final 0-100 risk score.
    Logic: Normalize all to 0-1, combine using weights, convert to 0-100, and clamp.
    """
    # 1. Normalize all components to 0-1 scale
    def normalize(val):
        if val > 1.0:
            return val / 100.0
        return val

    ml_val       = normalize(component_scores.get("ml", 0))
    behav_val    = normalize(component_scores.get("behavioral", 0))
    device_val   = normalize(component_scores.get("device", 0))
    graph_val    = normalize(component_scores.get("graph", 0))
    chain_val    = normalize(component_scores.get("chain", 0))

    # 2. Combine using provided weights
    # Normalize weights to sum to 1.0 (some are in 0-100 scale from API)
    w_sum = sum(weights.values()) or 100
    w_ml = weights.get("ml", 25) / w_sum
    w_be = weights.get("behavioral", 25) / w_sum
    w_de = weights.get("device", 15) / w_sum
    w_gr = weights.get("graph", 15) / w_sum
    w_ch = weights.get("chain", 20) / w_sum

    final_score = (
        ml_val * w_ml +
        behav_val * w_be +
        device_val * w_de +
        graph_val * w_gr +
        chain_val * w_ch
    )

    # 3. Convert back to 0-100 scale and 4. Clamp
    final_res = final_score * 100.0
    return max(0, min(100, final_res))


if __name__ == "__main__":
    # Test the scoring function with sample data
    print("\n" + "="*50)
    print("Testing scoring.py")
    print("="*50 + "\n")
    
    # Sample transaction data
    sample_tx = {
        'amount': 5000.0,
        'merchant_id': 'M001',
        'timestamp': datetime.now(),
        'oldbalanceOrg': 10000.0,
        'newbalanceOrig': 5000.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 5000.0,
        'tx_type': 'TRANSFER'
    }
    
    sample_profile = {
        'avg_transaction': 2500.0,
        'std_transaction': 800.0
    }
    
    # Score the transaction
    scores = score_transaction(sample_tx, sample_profile)
    
    print("Sample Transaction:")
    print(f"  Amount: ₹{sample_tx['amount']}")
    print(f"  Type: {sample_tx['tx_type']}")
    print(f"  Merchant: {sample_tx['merchant_id']}")
    print()
    
    print("ML Scores:")
    print(f"  Isolation Forest: {scores['if_score']:.2f}")
    print(f"  XGBoost:          {scores['xgb_score']:.2f}")
    print(f"  Ensemble (ML):    {scores['ml_score']:.2f}")
    print(f"  Risk Category:    {interpret_score(scores['ml_score'])}")
    print()
    
    # Show feature importance
    importance = get_feature_importance()
    if importance:
        print("Feature Importance (XGBoost):")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features:
            print(f"  {feature:25s}: {imp:.4f}")
    
    print()