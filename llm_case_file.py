import asyncio
import google.generativeai as genai
import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None


def get_similar_cases(fraud_type, db, limit=3):
    """
    Query similar past cases from the database.
    
    Args:
        fraud_type: Type of fraud to search for
        db: Database connection object
        limit: Maximum number of cases to retrieve
    
    Returns:
        list of case_file strings
    """
    cursor = db.cursor()
    cursor.execute(
        "SELECT case_file FROM cases WHERE fraud_type = ? ORDER BY created_at DESC LIMIT ?",
        (fraud_type, limit)
    )
    rows = cursor.fetchall()
    
    if not rows:
        return []
    
    return [row[0] for row in rows]


async def generate_case_file(tx_data, risk_score, flags, chain_state, db):
    """
    Generate an LLM-powered case file for a flagged transaction.
    
    Args:
        tx_data: dict with tx_id, user_id, amount, merchant_id, city, timestamp, etc.
        risk_score: Calculated risk score (0-100)
        flags: list of flag names
        chain_state: string like "CHAIN_ACTIVE" or "SAFE"
        db: Database connection object
    
    Returns:
        string containing the generated case file
    """
    tx_id = tx_data['tx_id']
    user_id = tx_data['user_id']
    amount = tx_data['amount']
    merchant_id = tx_data['merchant_id']
    city = tx_data.get('city', 'Unknown')
    timestamp = tx_data['timestamp']
    
    # FRAUD TYPE CLASSIFICATION
    if "CHAIN_ACTIVE" in chain_state or "LOCKED" in chain_state:
        fraud_type = "ATO"
    elif "CROSS_CHANNEL_BURST" in flags:
        fraud_type = "coordinated"
    elif "GRADUATED_ESCALATION" in flags or "JUST_BELOW_THRESHOLD" in flags:
        fraud_type = "stealth_probe"
    elif amount > 10000:
        fraud_type = "high_value"
    else:
        fraud_type = "general_anomaly"
    
    # RETRIEVE SIMILAR CASES
    similar_cases = get_similar_cases(fraud_type, db, limit=3)
    similar_cases_text = "\n".join([f"- {case}" for case in similar_cases]) if similar_cases else "No similar cases on record"
    
    # BUILD PROMPT
    prompt = f"""You are a fraud investigation analyst. Analyze this transaction and write a 4-sentence case file.

TRANSACTION DETAILS:
- ID: {tx_id}
- User: {user_id}
- Amount: ₹{amount:,.2f}
- Merchant: {merchant_id}
- Location: {city}
- Time: {timestamp}
- Risk Score: {risk_score}/100

RISK FLAGS:
{', '.join(flags) if flags else 'None'}

CHAIN STATE: {chain_state}

SIMILAR PAST CASES:
{similar_cases_text}

Write a concise 4-sentence investigation report covering:
1. What triggered the alert
2. Behavioral pattern assessment
3. Risk classification
4. Recommended action

Be specific and reference the transaction details."""
    
    # CALL GEMINI API
    try:
        if not model:
            raise Exception("Gemini API key not configured")
        
        response = await asyncio.to_thread(model.generate_content, prompt)
        case_file = response.text
        
        # Store in cases table
        db.execute(
            "INSERT INTO cases (tx_id, fraud_type, case_file, created_at) VALUES (?, ?, ?, ?)",
            (tx_id, fraud_type, case_file, datetime.now())
        )
        db.commit()
        
        return case_file
    
    except Exception as e:
        # FALLBACK: Load from case_cache/{fraud_type}_case.json
        cache_file = f"case_cache/{fraud_type}_case.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    return json.load(f)['case_file']
            except:
                pass
        return f"""**AUTOMATED FRAUD INTELLIGENCE REPORT**

1. **Trigger:** Transaction {tx_id} flagged by monitoring thresholds with risk score {risk_score}/100.
2. **Behavioral Assessment:** System detected {', '.join(flags) if flags else 'minor deviations'} anomalies indicating potential {fraud_type} behavior.
3. **Risk Profile:** Categorized as {'HIGH RISK' if risk_score >= 66 else 'ELEVATED RISK' if risk_score >= 36 else 'LOW RISK'} due to monitored indicators.
4. **Action:** Recommend {'immediate block' if risk_score >= 66 else 'MFA verification' if risk_score >= 36 else 'approval'}. Manual review may be required if disputed."""

async def classify_transfer_purpose(purpose_text):
    """
    Classify a UPI transfer purpose for APP scam detection.
    
    Args:
        purpose_text: User-entered transfer purpose description
    
    Returns:
        dict with classification, scam_type, and confidence
    """
    prompt = f"""Classify this UPI transfer purpose as one of:
- CONFIRMED_SCAM (romance scam, investment scam, impersonation)
- POSSIBLE_SCAM (suspicious but not certain)
- LEGITIMATE (normal payment)

Transfer purpose: "{purpose_text}"

Respond with JSON: {{"classification": "...", "scam_type": "...", "confidence": 0-100}}"""
    
    try:
        if not model:
            raise Exception("Gemini API key not configured")
        
        response = await asyncio.to_thread(model.generate_content, prompt)
        # Attempt to parse json from text
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    
    except:
        # FALLBACK: Keyword matching
        scam_keywords = [
            "investment", "crypto", "bitcoin", "urgent", "prize", 
            "lottery", "tax payment", "court", "arrest"
        ]
        
        if any(kw in purpose_text.lower() for kw in scam_keywords):
            return {
                "classification": "POSSIBLE_SCAM",
                "scam_type": "keyword_match",
                "confidence": 60
            }
        else:
            return {
                "classification": "LEGITIMATE",
                "scam_type": None,
                "confidence": 50
            }