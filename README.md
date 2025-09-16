# Course Recommendation System

This project implements an AI-based course recommendation system that matches user skill gaps, identified from JSON assessment data, with relevant courses stored in an Excel (XLSX) file. The system uses Python with libraries like pandas and scikit-learn to process data and provide personalized course recommendations.

## Overview

The application analyzes a user's performance in various sub-skills (e.g., IoT basics, Python for IoT) from a JSON assessment file and compares these against a catalog of courses in an XLSX file. It identifies skill gaps where the user's score falls below a predefined benchmark and recommends the top three courses that address those gaps using a content-based filtering approach with TF-IDF and cosine similarity.

## Features

- **Data Loading**: Reads user assessment data from JSON and course data from XLSX files.
- **Skill Gap Identification**: Detects sub-skills where the user performs below a configurable benchmark score (default: 70).
- **Recommendation Engine**: Uses TF-IDF vectorization and cosine similarity to match skill gaps with course content.
- **Output**: Provides a ranked list of the top three recommended courses with relevance scores.