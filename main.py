from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

app = FastAPI(title="Recommendation Microservice")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000","trelix-livid.vercel.app","https://trelix-xj5h.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
try:
    client = MongoClient(MONGODB_URI)
    db = client["Trelix"]
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Pydantic model for response
class Course(BaseModel):
    _id: str
    title: str
    typeRessource: str
    level: str
    price: float

class RecommendationResponse(BaseModel):
    user_id: str
    preferences: List[str]
    recommendations: List[Course]

# Load and preprocess data
def load_data():
    try:
        users_df = pd.DataFrame(list(db.users.find()))
        prefs_df = pd.DataFrame(list(db.preferences.find()))
        courses_df = pd.DataFrame(list(db.courses.find()))

        # Preprocess courses
        courses_df['typeRessource'] = courses_df['typeRessource'].fillna('other')
        courses_df['typeRessource'] = courses_df['typeRessource'].str.lower().str.strip()
        courses_df['_id'] = courses_df['_id'].astype(str)

        # Preprocess preferences
        prefs_df['typeRessource'] = prefs_df['typeRessource'].str.lower().str.strip()
        prefs_df['typeRessource'] = prefs_df['typeRessource'].replace(' interactive exercice', 'interactive exercice')
        prefs_df['user'] = prefs_df['user'].apply(str)

        logger.info("Data loaded and preprocessed successfully")
        return prefs_df, courses_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

# Get user preferences
def get_user_preferences(user_id: str, prefs_df: pd.DataFrame) -> List[str]:
    try:
        user_prefs = prefs_df[prefs_df['user'] == user_id]['typeRessource'].tolist()
        logger.info(f"User {user_id} preferences: {user_prefs}")
        return user_prefs if user_prefs else []
    except Exception as e:
        logger.error(f"Error getting preferences for user {user_id}: {str(e)}")
        return []

# Recommend courses
def recommend_courses(user_id: str, prefs_df: pd.DataFrame, courses_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    try:
        user_prefs = get_user_preferences(user_id, prefs_df)

        if not user_prefs:
            logger.info(f"No preferences found for user {user_id}")
            return []

        # Filter courses that match preferences
        matched_courses = courses_df[courses_df['typeRessource'].isin(user_prefs)]

        if matched_courses.empty:
            logger.info(f"No courses match preferences for user {user_id}")
            return []

        # Select top N courses
        recommended_courses = matched_courses[['_id', 'title', 'typeRessource', 'level', 'price']].head(top_n).to_dict(orient='records')

        logger.info(f"Recommended {len(recommended_courses)} courses for user {user_id}")
        return recommended_courses
    except Exception as e:
        logger.error(f"Error recommending courses for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error recommending courses: {str(e)}")

# API endpoint
@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, top_n: int = 10):
    try:
        if top_n < 1 or top_n > 50:
            raise HTTPException(status_code=400, detail="top_n must be between 1 and 50")

        # Load data
        prefs_df, courses_df = load_data()

        # Get recommendations
        recommendations = recommend_courses(user_id, prefs_df, courses_df, top_n)

        # Prepare response
        response = {
            "user_id": user_id,
            "preferences": get_user_preferences(user_id, prefs_df),
            "recommendations": recommendations
        }

        logger.info(f"Returning recommendations for user {user_id}: {len(recommendations)} courses")
        return response
    except Exception as e:
        logger.error(f"Error processing recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing recommendations: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # Render injects this PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)

