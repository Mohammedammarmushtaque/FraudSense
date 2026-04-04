import sqlite3
import json
from datetime import datetime, timedelta
from geopy.distance import geodesic

CITY_TIMEZONE_MAP = {
    "Mumbai": "Asia/Kolkata", "Delhi": "Asia/Kolkata", "Bangalore": "Asia/Kolkata",
    "Chennai": "Asia/Kolkata", "Kolkata": "Asia/Kolkata", "Hyderabad": "Asia/Kolkata",
    "Pune": "Asia/Kolkata", "Ahmedabad": "Asia/Kolkata",
    "Chicago": "America/Chicago",
    "New York": "America/New_York",
    "Seattle": "America/Los_Angeles",
    "Miami": "America/New_York",
    "Lagos": "Africa/Lagos"
}

CITY_COORDINATES = {
    "Mumbai": (19.0760, 72.8777), "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639), "Hyderabad": (17.3850, 78.4867)
}


def check_device(user_id, device_id, db):
    """Check if device is new for this user"""
    cursor = db.cursor()
    
    # Check if device exists for this user
    cursor.execute(
        "SELECT * FROM user_devices WHERE user_id=? AND device_id=?",
        (user_id, device_id)
    )
    device = cursor.fetchone()
    
    if device:
        # Update last_seen
        cursor.execute(
            "UPDATE user_devices SET last_seen=? WHERE user_id=? AND device_id=?",
            (datetime.now().isoformat(), user_id, device_id)
        )
        db.commit()
        return {"is_new": False, "risk_add": 0}
    else:
        # Insert new device
        cursor.execute(
            """INSERT INTO user_devices (user_id, device_id, city, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, device_id, "unknown", datetime.now().isoformat(), datetime.now().isoformat())
        )
        db.commit()
        return {
            "is_new": True,
            "risk_add": 20,
            "flag": "NEW_DEVICE",
            "chain_event": "LOGIN_NEW_DEVICE"
        }


def check_timezone_mismatch(user_id, device_timezone, city, db):
    """Check if device timezone matches expected timezone for city"""
    expected_tz = CITY_TIMEZONE_MAP.get(city, "Asia/Kolkata")
    
    if device_timezone != expected_tz:
        return {
            "flag": "VPN_SUSPECTED",
            "risk_add": 25,
            "message": f"Timezone mismatch: device reports {device_timezone} but location is {city}"
        }
    
    return {"flag": None, "risk_add": 0}


def check_impossible_travel(user_id, city, timestamp, db):
    """Check if user travel between cities is physically impossible"""
    cursor = db.cursor()
    
    # Get last transaction location and time
    cursor.execute(
        """SELECT city, timestamp FROM transactions 
           WHERE user_id=? ORDER BY timestamp DESC LIMIT 1""",
        (user_id,)
    )
    last_tx = cursor.fetchone()
    
    if last_tx:
        last_city, last_timestamp = last_tx
        
        # Parse timestamps safely regardless of timezone aware/naive
        if isinstance(timestamp, str):
            current_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
        else:
            current_time = timestamp.replace(tzinfo=None) if getattr(timestamp, 'tzinfo', None) else timestamp
            
        if isinstance(last_timestamp, str):
            last_time = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
        else:
            last_time = last_timestamp.replace(tzinfo=None) if getattr(last_timestamp, 'tzinfo', None) else last_timestamp
        
        # Calculate time difference in minutes
        time_diff = (current_time - last_time).total_seconds() / 60
        
        # Calculate distance if both cities have coordinates
        if city != last_city and city in CITY_COORDINATES and last_city in CITY_COORDINATES:
            coords1 = CITY_COORDINATES[last_city]
            coords2 = CITY_COORDINATES[city]
            distance_km = geodesic(coords1, coords2).kilometers
            
            # Minimum travel time assuming 500 km/h flight speed
            min_travel_time = (distance_km / 500) * 60
            
            if time_diff < min_travel_time:
                return {
                    "flag": "IMPOSSIBLE_TRAVEL",
                    "risk_add": 40,
                    "message": f"User in {last_city} {time_diff:.0f}min ago, now in {city} ({distance_km:.0f} km away)"
                }
    
    return {"flag": None, "risk_add": 0}


def check_velocity(user_id, timestamp, db):
    """Check transaction velocity - too many transactions in short time"""
    cursor = db.cursor()
    
    # Count transactions in last 60 minutes
    cursor.execute(
        """SELECT COUNT(*) FROM transactions 
           WHERE user_id=? AND timestamp > datetime('now', '-60 minutes')""",
        (user_id,)
    )
    count = cursor.fetchone()[0]
    
    if count > 10:
        return {
            "flag": "VELOCITY_BREACH",
            "risk_add": 30,
            "count": count
        }
    
    return {"flag": None, "risk_add": 0}


def check_cross_channel_burst(user_id, channel, timestamp, db):
    """Check if user is accessing account across multiple channels rapidly"""
    cursor = db.cursor()
    
    # Count distinct channels in last 10 minutes
    cursor.execute(
        """SELECT COUNT(DISTINCT channel) as channel_count 
           FROM channel_activity 
           WHERE user_id=? AND timestamp > datetime('now', '-10 minutes')""",
        (user_id,)
    )
    result = cursor.fetchone()
    channel_count = result[0] if result else 0
    
    if channel_count >= 3:
        return {
            "flag": "CROSS_CHANNEL_BURST",
            "risk_add": 30,
            "message": f"Account accessed across {channel_count} payment channels in 10 minutes"
        }
    
    return {"flag": None, "risk_add": 0}


def check_synthetic_identity(user_id, db):
    """Check for synthetic identity patterns"""
    cursor = db.cursor()
    
    # Query user profile stats
    cursor.execute(
        """SELECT 
            COUNT(*) as tx_count,
            AVG(amount) as avg_amount,
            (SELECT COUNT(DISTINCT merchant_id) FROM transactions WHERE user_id=?) as merchant_diversity,
            SUM(CASE WHEN decision='BLOCK' THEN 1 ELSE 0 END) as prior_blocks,
            MIN(timestamp) as first_seen
           FROM transactions WHERE user_id=?""",
        (user_id, user_id)
    )
    
    result = cursor.fetchone()
    
    if not result or result[0] == 0:
        # No transaction history, treat as not suspicious
        return {"flag": None, "risk_add": 0}
    
    tx_count, avg_amount, merchant_diversity, prior_blocks, first_seen = result
    
    # Handle None values
    avg_amount = avg_amount if avg_amount else 0
    merchant_diversity = merchant_diversity if merchant_diversity else 0
    prior_blocks = prior_blocks if prior_blocks else 0
    
    # Calculate account age in days
    if first_seen:
        if isinstance(first_seen, str):
            first_seen_dt = datetime.fromisoformat(first_seen.replace("Z", "+00:00")).replace(tzinfo=None)
        else:
            first_seen_dt = first_seen.replace(tzinfo=None) if getattr(first_seen, 'tzinfo', None) else first_seen
        account_age = (datetime.now() - first_seen_dt).days
    else:
        account_age = 0
    
    # Check 5 signals
    signals = 0
    
    if account_age < 90:
        signals += 1
    if merchant_diversity <= 2:
        signals += 1
    if prior_blocks == 0 and tx_count > 10:
        signals += 1
    
    if signals >= 3:
        return {
            "flag": "SYNTHETIC_IDENTITY_SUSPECTED",
            "risk_add": 35,
            "signals": signals,
            "high_confidence": signals >= 4
        }
    
    return {"flag": None, "risk_add": 0}