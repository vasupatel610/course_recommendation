# Enhanced Course Recommendation System

An intelligent course recommendation system that uses advanced machine learning techniques to provide personalized learning recommendations based on user skill assessments.

## 🚀 Features

### Core Capabilities
- **Hybrid Recommendation Engine**: Combines multiple recommendation strategies for optimal results
- **Semantic Search**: Uses SBERT (Sentence-BERT) embeddings for better understanding of course content
- **Skill Gap Analysis**: Identifies learning gaps from user assessment data
- **Progressive Learning**: Recommends courses based on difficulty progression
- **Domain-Based Filtering**: Groups courses by technology domains (IoT, Data Science, Web Development, etc.)
- **REST API**: FastAPI-based API for easy integration

### Recommendation Strategies
1. **Skill-Based Recommendations** (40% weight) - Addresses identified weak skills
2. **Domain-Based Recommendations** (30% weight) - Strengthens weak technology domains  
3. **Progressive Learning** (20% weight) - Follows natural learning progression
4. **Complementary Skills** (10% weight) - Suggests related technologies

## 🏗️ Architecture

```
├── api.py                 # FastAPI REST API server
├── recommend.py           # Core recommendation engine
├── data/
│   ├── courses.ods       # Course catalog (ODS/XLSX format)
│   └── assessments/
│       └── user_assessment.json  # User skill assessment data
```

## 📋 Requirements

### Dependencies
```bash
# Core ML and Data Processing
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0

# API Framework
fastapi>=0.68.0
uvicorn>=0.15.0

# File Processing
openpyxl>=3.0.0  # For XLSX files
odfpy>=1.4.0     # For ODS files

# Utilities
pathlib
logging
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd course-recommendation-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data files**
   - Place course data in `data/courses.ods` (or `.xlsx`)
   - Place assessment data in `data/assessments/user_assessment.json`

## 🔧 Usage

### Starting the API Server

```bash
python api.py
```

The server will start on `http://localhost:8000`

### API Documentation

Access interactive API docs at: `http://localhost:8000/docs`

### Making Recommendations

**Endpoint:** `POST /recommend`

**Request Body:**
```json
[
  {
    "assessment_result": {
      "skills": [
        {
          "sub_skills": [
            {
              "sub_skill_code": "python-basics",
              "score": 45,
              "Weightage": 2
            },
            {
              "sub_skill_code": "data-analysis",
              "score": 65,
              "Weightage": 3
            }
          ]
        }
      ]
    }
  }
]
```

**Response:**
```json
[
  {
    "CourseID": "CS101",
    "CourseTitle": "Python Programming Fundamentals",
    "Domain": "Programming",
    "Difficulty": "Beginner",
    "RelevanceScore": 0.875,
    "Reason": "Addresses your weak skills; Strengthens your programming domain",
    "ExpectedImpact": "4.2"
  }
]
```

### Standalone Usage

```python
from recommend import EnhancedCourseRecommender

# Initialize recommender
recommender = EnhancedCourseRecommender(use_semantic_embeddings=True)

# Load course data
recommender.load_course_data("data/courses.ods")

# Load assessment data
skill_profile = recommender.load_assessment_data("data/assessments/user_assessment.json")

# Generate recommendations
recommendations = recommender.recommend_courses_hybrid(skill_profile, top_n=5)
print(recommendations)
```

## 📊 Data Formats

### Course Data Structure (ODS/XLSX)
Required columns:
- `Course ID`: Unique identifier
- `Course Title`: Name of the course
- `Domain`: Technology domain (IoT, Data Science, etc.)
- `Skill Areas`: Comma-separated skills covered
- `Category`: Course category (optional)

### Assessment Data Structure (JSON)
```json
[
  {
    "assessment_result": {
      "skills": [
        {
          "sub_skills": [
            {
              "sub_skill_code": "skill-name",
              "score": 0-100,
              "Weightage": 1-5
            }
          ]
        }
      ]
    }
  }
]
```

## 🎯 How It Works

### 1. Skill Analysis
- Analyzes user assessment scores against benchmark (default: 70%)
- Identifies weak skills requiring improvement
- Maps skills to technology domains
- Calculates overall performance metrics

### 2. Course Matching
- **Semantic Matching**: Uses SBERT to understand course content semantically
- **TF-IDF Fallback**: Traditional text matching for robustness
- **Domain Filtering**: Prioritizes courses in weak domains
- **Difficulty Progression**: Suggests appropriate difficulty levels

### 3. Recommendation Scoring
- Combines multiple strategies with weighted scores
- Considers skill priority (high/medium based on score)
- Factors in learning progression
- Includes complementary skill suggestions

### 4. Result Ranking
- Sorts by relevance score
- Provides explanation for each recommendation
- Estimates expected learning impact (1-5 scale)