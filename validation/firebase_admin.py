import firebase_admin

from firebase_admin import credentials, auth as firebase_auth
from fastapi import Header, HTTPException

# initialise firebase admin
cred = credentials.Certificate("validation/fyp-fab-firebase-adminsdk-fbsvc-d18177c13c.json")
firebase_admin.initialize_app(cred)

# token validation
async def verify_token(authorization: str = Header(...)):
    try:
        token = authorization.split("Bearer ")[-1]
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")