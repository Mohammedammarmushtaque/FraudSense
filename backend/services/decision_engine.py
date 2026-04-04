"""
FraudSense Decision Maker Module
================================

Production-grade final decision engine for fraud detection.
Integrates with RiskEngine to provide multi-layer fraud protection
with explainable outputs.

Author: FraudSense Engineering Team
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class Decision(Enum):
    """Final decision enumeration."""
    APPROVE = "APPROVE"
    MFA_HOLD = "MFA_HOLD"
    BLOCK = "BLOCK"


class ConfidenceLevel(Enum):
    """Confidence level enumeration."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DecisionEngine:
    """
    Core decision engine class for transaction evaluation.
    Provides logic for classifying risk scores into decisions.
    """
    @staticmethod
    def decide(raw_score: float, reasons: list, component_scores: dict) -> dict:
        """
        Bridge method between RiskEngine raw scores and API decision requirements.
        """
        # 1. Base classification matching RiskEngine's 75/45 rules
        if raw_score >= 75:
            decision = "BLOCK"
            severity = "CRITICAL"
        elif raw_score >= 45:
            decision = "MFA_HOLD"
            severity = "ELEVATED"
        else:
            decision = "APPROVE"
            severity = "NORMAL"

        # 2. Extract ML details with safe defaults
        ml_score = component_scores.get("ml", 0.0)
        ml_prob = component_scores.get("_fraud_probability", ml_score / 100.0)

        # 3. Format result for transaction API
        return {
            "decision":    decision,
            "risk_score":  round(float(raw_score) / 100.0, 4),  # normalized 0-1
            "final_score": float(raw_score),
            "severity":    severity,
            "reasons":     reasons,
            "ml_score":    round(float(ml_score), 2),
            "rule_score":  round(float(raw_score), 2),
            "explanation": f"Pipeline analysis complete. Integrated risk score: {raw_score}/100.",
        }


@dataclass
class DecisionConfig:
    """
    Configuration for decision thresholds.
    
    All thresholds can be customized for different environments
    or regulatory requirements.
    """
    # Base thresholds
    BLOCK_THRESHOLD: int = 75
    MFA_THRESHOLD: int = 45
    
    # Dynamic threshold adjustments
    NIGHT_DEVICE_BLOCK_ADJUSTMENT: int = 10
    BEHAVIORAL_BLOCK_ADJUSTMENT: int = 5
    
    # Critical fraud pattern thresholds
    CRITICAL_AMOUNT_THRESHOLD: int = 80
    CRITICAL_DEVICE_THRESHOLD: int = 50
    CRITICAL_AMOUNT_LOWER: int = 70
    CRITICAL_LOCATION_THRESHOLD: int = 60
    CRITICAL_BEHAVIORAL_THRESHOLD: int = 85
    CRITICAL_GRAPH_THRESHOLD: int = 75
    CRITICAL_CHAIN_THRESHOLD: int = 75
    
    # Multi-layer escalation thresholds
    MULTI_LAYER_COMPONENT_THRESHOLD: int = 60
    MULTI_LAYER_MIN_COMPONENTS: int = 3
    
    # High-risk merchant configuration
    HIGH_RISK_CATEGORIES: List[str] = field(default_factory=lambda: [
        "gambling", "casino", "betting", "crypto", "forex"
    ])
    HIGH_RISK_MERCHANT_MIN_RISK: int = 60
    
    # ML override thresholds
    ML_OVERRIDE_SCORE: int = 85
    ML_OVERRIDE_RISK_SCORE: int = 60
    
    # False negative prevention
    FALSE_NEGATIVE_RISK_THRESHOLD: int = 65


# Default configuration singleton
DEFAULT_CONFIG = DecisionConfig()


# =============================================================================
# DEFENSIVE HELPER FUNCTIONS
# =============================================================================

def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Handles None inputs, missing keys, and non-dict types.
    """
    if data is None:
        return default
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default


def _safe_get_component(
    scores: Any, 
    component: str, 
    default: float = 0.0
) -> float:
    """
    Safely get a component score with type coercion.
    
    Returns default if any error occurs during extraction or conversion.
    """
    if scores is None:
        return default
    try:
        value = scores.get(component, default)
        if value is None:
            return default
        return float(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _prepend_reason(reasons: Optional[List[str]], reason: str) -> List[str]:
    """Prepend a reason to the reasons list, handling None input."""
    if reasons is None:
        reasons = []
    reasons.insert(0, reason)
    return reasons


def _get_top_components(
    component_scores: Dict[str, float], 
    top_n: int = 3
) -> List[str]:
    """
    Get the top N highest scoring components.
    
    Returns empty list if component_scores is invalid.
    """
    if not component_scores or not isinstance(component_scores, dict):
        return []
    try:
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1] if x[1] is not None else 0.0,
            reverse=True
        )
        return [comp for comp, _ in sorted_components[:top_n]]
    except (AttributeError, TypeError):
        return []


# =============================================================================
# FRAUD PATTERN DETECTION FUNCTIONS
# =============================================================================

def _is_high_risk_merchant(tx_data: Dict[str, Any], config: DecisionConfig) -> bool:
    """
    Check if transaction is to a high-risk merchant category.
    
    Examines multiple merchant-related fields for risk category keywords.
    """
    if not tx_data or not isinstance(tx_data, dict):
        return False
    
    try:
        # Check multiple possible field names for merchant category
        merchant_category = str(_safe_get(tx_data, "merchant_category", "")).lower()
        merchant_type = str(_safe_get(tx_data, "merchant_type", "")).lower()
        category_code = str(_safe_get(tx_data, "category_code", "")).lower()
        mcc = str(_safe_get(tx_data, "mcc", "")).lower()
        description = str(_safe_get(tx_data, "description", "")).lower()
        
        # Combine all text for comprehensive matching
        combined_text = " ".join([
            merchant_category, 
            merchant_type, 
            category_code, 
            mcc, 
            description
        ])
        
        for risky_category in config.HIGH_RISK_CATEGORIES:
            if risky_category in combined_text:
                logger.debug(f"High-risk merchant detected: {risky_category}")
                return True
        
        return False
    except Exception:
        return False


def _count_high_risk_components(
    component_scores: Dict[str, float], 
    threshold: int
) -> int:
    """
    Count how many components exceed the given threshold.
    
    Returns 0 if component_scores is invalid.
    """
    if not component_scores or not isinstance(component_scores, dict):
        return 0
    
    count = 0
    for component, score in component_scores.items():
        try:
            if score is not None and float(score) >= threshold:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def _detect_critical_patterns(
    component_scores: Dict[str, float],
    config: DecisionConfig
) -> List[str]:
    """
    Detect critical fraud patterns across components.
    
    Returns list of triggered pattern descriptions.
    """
    triggered_patterns = []
    
    # Extract all relevant component scores
    amount_score = _safe_get_component(component_scores, "amount")
    location_score = _safe_get_component(component_scores, "location")
    device_score = _safe_get_component(component_scores, "device")
    behavioral_score = _safe_get_component(component_scores, "behavioral")
    graph_score = _safe_get_component(component_scores, "graph")
    chain_score = _safe_get_component(component_scores, "chain")
    
    # Pattern 1: High amount + suspicious device
    if (amount_score >= config.CRITICAL_AMOUNT_THRESHOLD and 
            device_score >= config.CRITICAL_DEVICE_THRESHOLD):
        triggered_patterns.append(
            f"High amount ({amount_score:.0f}) + suspicious device ({device_score:.0f})"
        )
    
    # Pattern 2: Elevated amount + location mismatch
    if (amount_score >= config.CRITICAL_AMOUNT_LOWER and 
            location_score >= config.CRITICAL_LOCATION_THRESHOLD):
        triggered_patterns.append(
            f"Elevated amount ({amount_score:.0f}) + location risk ({location_score:.0f})"
        )
    
    # Pattern 3: Extreme behavioral anomaly
    if behavioral_score >= config.CRITICAL_BEHAVIORAL_THRESHOLD:
        triggered_patterns.append(
            f"Extreme behavioral anomaly ({behavioral_score:.0f})"
        )
    
    # Pattern 4: Graph fraud indicators
    if graph_score >= config.CRITICAL_GRAPH_THRESHOLD:
        triggered_patterns.append(
            f"Graph fraud indicators ({graph_score:.0f})"
        )
    
    # Pattern 5: Chain analysis fraud signals
    if chain_score >= config.CRITICAL_CHAIN_THRESHOLD:
        triggered_patterns.append(
            f"Chain analysis fraud signals ({chain_score:.0f})"
        )
    
    return triggered_patterns


# =============================================================================
# CONFIDENCE CALCULATION
# =============================================================================

def _calculate_confidence(risk_score: int) -> str:
    """
    Calculate confidence level based on risk score.
    
    Higher risk scores indicate higher confidence in fraud assessment.
    """
    if risk_score > 75:
        return ConfidenceLevel.HIGH.value
    elif risk_score > 45:
        return ConfidenceLevel.MEDIUM.value
    else:
        return ConfidenceLevel.LOW.value


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

def _print_debug_output(
    risk_score: int,
    base_decision: str,
    final_decision: str,
    high_risk_count: int
) -> None:
    """Print debug output for demonstration and debugging."""
    print("=" * 30)
    print("FINAL DECISION ENGINE")
    print(f"Risk Score: {risk_score}")
    print(f"Base Decision: {base_decision}")
    print(f"Final Decision: {final_decision}")
    print(f"High Risk Components: {high_risk_count}")
    print("=" * 30)


# =============================================================================
# MAIN DECISION FUNCTION
# =============================================================================

def make_decision(
    user_id: str,
    tx_data: dict,
    risk_engine,
    config: DecisionConfig = None
) -> dict:
    """
    Make final fraud decision with multi-layer protection.
    
    This function integrates with the RiskEngine to provide:
    - Context-aware decision logic
    - Multi-layer escalation rules  
    - Dynamic thresholding
    - Fraud pattern recognition
    - Strong explainability
    
    Args:
        user_id: Unique identifier for the user
        tx_data: Transaction data dictionary containing:
            - amount: Transaction amount
            - merchant_category: Merchant category code/name
            - is_night: 1 if transaction at night, 0 otherwise
            - And other transaction fields
        risk_engine: RiskEngine instance with calculate_risk method
        config: Optional DecisionConfig for custom thresholds
    
    Returns:
        Dictionary containing:
        - decision: Final decision ("APPROVE", "MFA_HOLD", "BLOCK")
        - risk_score: Overall risk score (0-100)
        - confidence: Confidence level ("HIGH", "MEDIUM", "LOW")
        - reasons: List of explainable reasons for the decision
        - component_scores: Individual component risk scores
        - rule_score: Rule-based risk score
        - ml_score: ML-based risk score
    
    Example:
        >>> result = make_decision(
        ...     user_id="user_123",
        ...     tx_data={"amount": 5000, "merchant_category": "retail"},
        ...     risk_engine=risk_engine
        ... )
        >>> print(result["decision"])
        'APPROVE'
    """
    # =========================================================================
    # INITIALIZATION & CONFIGURATION
    # =========================================================================
    if config is None:
        config = DEFAULT_CONFIG
    
    # Validate user_id
    if not user_id or not isinstance(user_id, str):
        user_id = "UNKNOWN_USER"
    
    # Validate tx_data
    if not tx_data or not isinstance(tx_data, dict):
        tx_data = {}
    
    # =========================================================================
    # STEP 1: GET RISK ENGINE OUTPUT
    # =========================================================================
    try:
        result = risk_engine.calculate_risk(user_id, tx_data)
    except Exception as e:
        logger.error(
            f"RiskEngine failure for user {user_id}: {type(e).__name__}: {e}"
        )
        # Fail-safe: block on engine failure to prevent fraud leakage
        return {
            "decision": Decision.BLOCK.value,
            "risk_score": 100,
            "confidence": ConfidenceLevel.HIGH.value,
            "reasons": [
                "RiskEngine failure - transaction blocked as safety measure",
                "Final Risk Score: 100/100"
            ],
            "component_scores": {},
            "rule_score": 100.0,
            "ml_score": 100.0
        }
    
    # Validate result
    if not result or not isinstance(result, dict):
        logger.error(f"Invalid RiskEngine result for user {user_id}")
        return {
            "decision": Decision.BLOCK.value,
            "risk_score": 100,
            "confidence": ConfidenceLevel.HIGH.value,
            "reasons": [
                "Invalid RiskEngine response - transaction blocked as safety measure",
                "Final Risk Score: 100/100"
            ],
            "component_scores": {},
            "rule_score": 100.0,
            "ml_score": 100.0
        }
    
    # =========================================================================
    # EXTRACT RISK ENGINE OUTPUTS WITH DEFENSIVE CODING
    # =========================================================================
    risk_score = int(_safe_get(result, "risk_score", 0))
    base_decision = str(_safe_get(result, "decision", "APPROVE")).upper()
    component_scores = _safe_get(result, "component_scores", {})
    reasons = _safe_get(result, "reasons", [])
    rule_score = float(_safe_get(result, "rule_score", 0.0))
    ml_score = float(_safe_get(result, "ml_score", 0.0))
    
    # Ensure reasons is a list
    if not isinstance(reasons, list):
        reasons = []
    
    # Ensure component_scores is a dict
    if not isinstance(component_scores, dict):
        component_scores = {}
    
    # Clamp risk_score to valid range
    risk_score = max(0, min(100, risk_score))
    
    # =========================================================================
    # STEP 2: INITIALIZE FINAL DECISION
    # =========================================================================
    final_decision = base_decision
    
    # =========================================================================
    # STEP 3: HARD SAFETY RULES (CRITICAL)
    # =========================================================================
    
    # -----------------------------------------------------------------
    # 3.1: NEVER DOWNGRADE BLOCK
    # -----------------------------------------------------------------
    if base_decision == Decision.BLOCK.value:
        final_decision = Decision.BLOCK.value
        logger.info(
            f"Maintaining BLOCK decision from RiskEngine for user {user_id}"
        )
    
    # -----------------------------------------------------------------
    # 3.2: CRITICAL MULTI-SIGNAL FRAUD DETECTION
    # -----------------------------------------------------------------
    critical_patterns = _detect_critical_patterns(component_scores, config)
    
    if critical_patterns:
        final_decision = Decision.BLOCK.value
        reasons = _prepend_reason(
            reasons,
            "Critical fraud pattern detected across multiple signals"
        )
        logger.warning(
            f"Critical fraud patterns for user {user_id}: {critical_patterns}"
        )
    
    # -----------------------------------------------------------------
    # 3.3: HIGH-RISK MERCHANT ESCALATION
    # -----------------------------------------------------------------
    if _is_high_risk_merchant(tx_data, config):
        if risk_score >= config.HIGH_RISK_MERCHANT_MIN_RISK:
            final_decision = Decision.BLOCK.value
            reasons = _prepend_reason(
                reasons,
                "High-risk merchant with elevated transaction risk"
            )
            logger.warning(
                f"High-risk merchant block for user {user_id} "
                f"with risk_score {risk_score}"
            )
    
    # -----------------------------------------------------------------
    # 3.4: MULTI-LAYER FRAUD ESCALATION
    # -----------------------------------------------------------------
    high_risk_component_count = _count_high_risk_components(
        component_scores,
        config.MULTI_LAYER_COMPONENT_THRESHOLD
    )
    
    if high_risk_component_count >= config.MULTI_LAYER_MIN_COMPONENTS:
        final_decision = Decision.BLOCK.value
        reasons = _prepend_reason(
            reasons,
            "Multiple high-risk layers detected"
        )
        logger.warning(
            f"Multi-layer escalation for user {user_id}: "
            f"{high_risk_component_count} components above "
            f"{config.MULTI_LAYER_COMPONENT_THRESHOLD} threshold"
        )
    
    # =========================================================================
    # STEP 4: DYNAMIC THRESHOLDS
    # =========================================================================
    block_threshold = config.BLOCK_THRESHOLD
    mfa_threshold = config.MFA_THRESHOLD
    
    # Extract relevant context from tx_data
    is_night = bool(_safe_get(tx_data, "is_night", 0))
    device_score = _safe_get_component(component_scores, "device")
    behavioral_score = _safe_get_component(component_scores, "behavioral")
    
    # Night + device risk adjustment
    if is_night and device_score > 40:
        block_threshold -= config.NIGHT_DEVICE_BLOCK_ADJUSTMENT
        logger.debug(
            f"Night+device adjustment: block_threshold -> {block_threshold}"
        )
    
    # Behavioral risk adjustment
    if behavioral_score > 70:
        block_threshold -= config.BEHAVIORAL_BLOCK_ADJUSTMENT
        logger.debug(
            f"Behavioral adjustment: block_threshold -> {block_threshold}"
        )
    
    # Ensure thresholds stay within valid bounds [0, 100]
    block_threshold = max(0, min(100, block_threshold))
    mfa_threshold = max(0, min(100, mfa_threshold))
    
    # Ensure MFA threshold doesn't exceed block threshold
    if mfa_threshold >= block_threshold:
        mfa_threshold = max(0, block_threshold - 1)
    
    # =========================================================================
    # STEP 5: THRESHOLD DECISION
    # =========================================================================
    if final_decision != Decision.BLOCK.value:
        if risk_score >= block_threshold:
            final_decision = Decision.BLOCK.value
        elif risk_score >= mfa_threshold:
            final_decision = Decision.MFA_HOLD.value
        else:
            final_decision = Decision.APPROVE.value
    
    # =========================================================================
    # STEP 6: ML OVERRIDE
    # =========================================================================
    if ml_score > config.ML_OVERRIDE_SCORE and risk_score > config.ML_OVERRIDE_RISK_SCORE:
        final_decision = Decision.BLOCK.value
        reasons = _prepend_reason(
            reasons,
            "ML model indicates extreme fraud probability"
        )
        logger.warning(
            f"ML override for user {user_id}: "
            f"ml_score={ml_score:.1f}, risk_score={risk_score}"
        )
    
    # =========================================================================
    # STEP 7: FALSE NEGATIVE PREVENTION
    # =========================================================================
    if (risk_score > config.FALSE_NEGATIVE_RISK_THRESHOLD and 
            final_decision == Decision.APPROVE.value):
        final_decision = Decision.MFA_HOLD.value
        reasons = _prepend_reason(
            reasons,
            "Preventive MFA due to elevated risk"
        )
        logger.info(
            f"False negative prevention MFA for user {user_id}: "
            f"risk_score={risk_score}"
        )
    
    # =========================================================================
    # STEP 8: EXPLAINABILITY ENGINE
    # =========================================================================
    
    # Add final risk score as a reason
    reasons.append(f"Final Risk Score: {risk_score}/100")
    
    # Add top contributing risk factors
    top_components = _get_top_components(component_scores, top_n=3)
    if top_components:
        top_factors_str = ", ".join(top_components)
        reasons.append(f"Top contributing risk factors: {top_factors_str}")
    
    # =========================================================================
    # STEP 9: CONFIDENCE SCORE
    # =========================================================================
    confidence_level = _calculate_confidence(risk_score)
    
    # =========================================================================
    # STEP 10: FINAL DEFENSIVE VALIDATION
    # =========================================================================
    
    # Validate final decision is a valid enum value
    valid_decisions = {d.value for d in Decision}
    if final_decision not in valid_decisions:
        logger.error(
            f"Invalid decision '{final_decision}' for user {user_id}, "
            f"defaulting to BLOCK"
        )
        final_decision = Decision.BLOCK.value
        reasons = _prepend_reason(
            reasons,
            "Invalid decision state - defaulted to BLOCK for safety"
        )
    
    # Final risk score validation (should already be clamped, but be safe)
    risk_score = max(0, min(100, risk_score))
    
    # =========================================================================
    # STEP 11: DEBUG OUTPUT
    # =========================================================================
    _print_debug_output(
        risk_score=risk_score,
        base_decision=base_decision,
        final_decision=final_decision,
        high_risk_count=high_risk_component_count
    )
    
    # =========================================================================
    # STEP 12: FINAL RETURN FORMAT
    # =========================================================================
    return {
        "decision": final_decision,
        "risk_score": risk_score,
        "confidence": confidence_level,
        "reasons": reasons,
        "component_scores": component_scores,
        "rule_score": rule_score,
        "ml_score": ml_score
    }


# =============================================================================
# EXTENDED FUNCTIONALITY FOR PRODUCTION USE
# =============================================================================

def make_decision_with_context(
    user_id: str,
    tx_data: dict,
    risk_engine,
    user_history: Optional[dict] = None,
    session_context: Optional[dict] = None,
    config: DecisionConfig = None
) -> dict:
    """
    Extended decision function with user history and session context.
    
    Args:
        user_id: Unique user identifier
        tx_data: Transaction data
        risk_engine: RiskEngine instance
        user_history: Optional dict containing:
            - prior_fraud_flags: int
            - account_age_days: int
            - avg_transaction_amount: float
            - mfa_completion_rate: float
        session_context: Optional dict containing:
            - session_duration_minutes: int
            - pages_visited: int
            - has_prior_mfa_in_session: bool
        config: Optional DecisionConfig
    
    Returns:
        Same format as make_decision with additional context-based reasons
    """
    # Get base decision
    result = make_decision(user_id, tx_data, risk_engine, config)
    
    # Apply context-based enhancements if history provided
    if user_history:
        prior_fraud_flags = int(_safe_get(user_history, "prior_fraud_flags", 0))
        account_age_days = int(_safe_get(user_history, "account_age_days", 365))
        avg_txn_amount = float(_safe_get(user_history, "avg_transaction_amount", 0))
        
        # New account + high risk escalation
        if account_age_days < 30 and result["risk_score"] > 50:
            if result["decision"] != Decision.BLOCK.value:
                result["decision"] = Decision.MFA_HOLD.value
                result["reasons"].insert(0, "New account with elevated risk - MFA required")
        
        # Prior fraud history escalation
        if prior_fraud_flags > 0 and result["risk_score"] > 40:
            result["decision"] = Decision.BLOCK.value
            result["reasons"].insert(0, f"Prior fraud history ({prior_fraud_flags} flags) with current risk")
        
        # Amount deviation check
        current_amount = float(_safe_get(tx_data, "amount", 0))
        if avg_txn_amount > 0 and current_amount > avg_txn_amount * 5:
            if result["decision"] == Decision.APPROVE.value:
                result["decision"] = Decision.MFA_HOLD.value
                result["reasons"].insert(0, "Transaction amount significantly exceeds historical average")
    
    # Apply session context if provided
    if session_context:
        has_prior_mfa = bool(_safe_get(session_context, "has_prior_mfa_in_session", False))
        
        # If already MFA'd in session and still flagged, consider blocking
        if has_prior_mfa and result["decision"] == Decision.MFA_HOLD.value:
            if result["risk_score"] > 60:
                result["decision"] = Decision.BLOCK.value
                result["reasons"].insert(0, "MFA already completed in session - escalating to block")
    
    return result


def batch_make_decisions(
    user_tx_pairs: List[tuple],
    risk_engine,
    config: DecisionConfig = None
) -> List[dict]:
    """
    Process multiple transactions in batch.
    
    Args:
        user_tx_pairs: List of (user_id, tx_data) tuples
        risk_engine: RiskEngine instance
        config: Optional DecisionConfig
    
    Returns:
        List of decision results in same order as input
    """
    results = []
    
    for idx, (user_id, tx_data) in enumerate(user_tx_pairs):
        try:
            result = make_decision(user_id, tx_data, risk_engine, config)
            results.append(result)
        except Exception as e:
            logger.error(
                f"Batch processing failed at index {idx} for user {user_id}: {e}"
            )
            results.append({
                "decision": Decision.BLOCK.value,
                "risk_score": 100,
                "confidence": ConfidenceLevel.HIGH.value,
                "reasons": [f"Batch processing error: {str(e)}"],
                "component_scores": {},
                "rule_score": 100.0,
                "ml_score": 100.0
            })
    
    return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "make_decision",
    "make_decision_with_context",
    "batch_make_decisions",
    "Decision",
    "ConfidenceLevel",
    "DecisionConfig",
    "DEFAULT_CONFIG",
]