from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any
from recommend import EnhancedCourseRecommender
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Course Recommendation API")

# Initialize recommender
recommender = EnhancedCourseRecommender(use_semantic_embeddings=True)
COURSE_FILE = "/home/artisans15/projects/course_recommendation/data/courses.ods"
recommender.load_course_data(COURSE_FILE)

@app.post("/recommend")
def recommend_courses(assessment: List[Dict[str, Any]] = Body(...)):
    try:
        # Directly pass to recommender
        skill_profile = recommender._analyze_assessment(assessment)
        recommendations = recommender.recommend_courses_hybrid(skill_profile, top_n=5)

        results = [
            {
                "CourseID": row["Course ID"],
                "CourseTitle": row["Course Title"],
                "Domain": row["Domain"],
                "Difficulty": row["Difficulty"],
                "RelevanceScore": row["Relevance Score"],
                "Reason": row["Recommendation Reason"],
                "ExpectedImpact": row["Expected Impact"],
            }
            for _, row in recommendations.iterrows()
        ]
        return results
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)