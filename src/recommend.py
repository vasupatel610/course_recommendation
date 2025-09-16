import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCourseRecommender:
    def __init__(self, use_semantic_embeddings=True):
        """
        Initialize the Enhanced Course Recommendation System
        
        Args:
            use_semantic_embeddings (bool): Whether to use semantic embeddings for better understanding
        """
        self.use_semantic_embeddings = use_semantic_embeddings
        self.model = None
        if use_semantic_embeddings:
            try:
                # Using a lightweight sentence transformer model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model for semantic embeddings")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}. Falling back to TF-IDF")
                self.use_semantic_embeddings = False
        
        self.scaler = StandardScaler()
        self.course_df = None
        self.skill_hierarchy = self._build_skill_hierarchy()
    
    def _build_skill_hierarchy(self):
        """Build a hierarchy of skills and their related domains"""
        return {
            'programming': ['python', 'java', 'javascript', 'c++', 'coding', 'software', 'development'],
            'data_science': ['data', 'analytics', 'statistics', 'machine learning', 'ai', 'visualization'],
            'web_development': ['html', 'css', 'react', 'frontend', 'backend', 'web'],
            'iot': ['iot', 'internet of things', 'sensors', 'embedded', 'arduino', 'raspberry'],
            'cloud': ['aws', 'azure', 'cloud', 'devops', 'docker', 'kubernetes'],
            'cybersecurity': ['security', 'cyber', 'encryption', 'network security', 'ethical hacking'],
            'mobile': ['android', 'ios', 'mobile', 'app development', 'flutter', 'react native'],
            'database': ['sql', 'database', 'mongodb', 'mysql', 'data management']
        }
    
    def load_course_data(self, file_path):
        """Enhanced course data loading with better preprocessing"""
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.xlsx':
                self.course_df = pd.read_excel(file_path, engine='openpyxl')
            elif file_ext == '.ods':
                self.course_df = pd.read_excel(file_path, engine='odf')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Clean and preprocess course data
            self.course_df = self._preprocess_course_data()
            logger.info(f"Loaded {len(self.course_df)} courses")
            return self.course_df
            
        except Exception as e:
            logger.error(f"Error loading course data: {e}")
            raise
    
    def _preprocess_course_data(self):
        """Enhanced preprocessing with better text representation"""
        df = self.course_df.copy() # type: ignore
        
        # Fill missing values
        df['Skill Areas'] = df['Skill Areas'].fillna('')
        df['Domain'] = df['Domain'].fillna('General')
        
        # Create enhanced text representation
        df['enhanced_text'] = df.apply(self._create_enhanced_text, axis=1)
        
        # Extract difficulty level if available
        df['difficulty'] = df.apply(self._extract_difficulty, axis=1)
        
        # Create course embeddings
        if self.use_semantic_embeddings:
            df['embeddings'] = self._create_semantic_embeddings(df['enhanced_text'].tolist())
        
        return df
    
    def _create_enhanced_text(self, row):
        """Create better text representation for each course"""
        components = []
        
        # Course title (highest weight)
        components.extend([row['Course Title'].lower()] * 3)
        
        # Domain (high weight)
        components.extend([row['Domain'].lower()] * 2)
        
        # Skill areas
        if pd.notna(row['Skill Areas']) and row['Skill Areas']:
            skills = re.split(r'[,;|]', str(row['Skill Areas']))
            components.extend([skill.strip().lower() for skill in skills])
        
        # Category
        if 'Category' in row and pd.notna(row['Category']):
            components.append(row['Category'].lower())
        
        return ' '.join(components)
    
    def _extract_difficulty(self, row):
        """Extract difficulty level from course title or description"""
        title = row['Course Title'].lower()
        if any(word in title for word in ['beginner', 'basic', 'intro', 'fundamentals']):
            return 1
        elif any(word in title for word in ['intermediate', 'advanced basics']):
            return 2
        elif any(word in title for word in ['advanced', 'expert', 'professional']):
            return 3
        return 2  # Default to intermediate
    
    def _create_semantic_embeddings(self, texts):
        """Create semantic embeddings for better understanding"""
        if self.model:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        return None
    
    def load_assessment_data(self, file_path):
        """Load and parse user assessment with enhanced analysis"""
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return self._analyze_assessment(data)
        except Exception as e:
            logger.error(f"Error loading assessment data: {e}")
            raise
    
    # def _analyze_assessment(self, assessment_data):
    #     """Enhanced assessment analysis with skill profiling"""
    #     user_skills = assessment_data[0]['assessment_result']['skills'][0]['sub_skills']
        
    #     skill_profile = {
    #         'weak_skills': [],
    #         'strong_skills': [],
    #         'skill_distribution': {},
    #         'overall_performance': 0,
    #         'domain_strengths': defaultdict(list),
    #         'domain_weaknesses': defaultdict(list)
    #     }
        
    #     total_score = 0
    #     skill_count = 0
        
    #     for skill in user_skills:
    #         score = skill['score']
    #         skill_name = skill['sub_skill_code'].lower().strip()
    #         weightage = skill.get('Weightage', 1)
            
    #         total_score += score
    #         skill_count += 1
            
    #         # Categorize skills based on performance
    #         if score < 60:  # Poor performance
    #             skill_profile['weak_skills'].append({
    #                 'name': skill_name,
    #                 'score': score,
    #                 'weightage': weightage,
    #                 'priority': 'high' if score < 40 else 'medium',
    #                 'domain': self._map_skill_to_domain(skill_name)
    #             })
    #         elif score >= 80:  # Strong performance
    #             skill_profile['strong_skills'].append({
    #                 'name': skill_name,
    #                 'score': score,
    #                 'domain': self._map_skill_to_domain(skill_name)
    #             })
            
    #         # Domain-wise analysis
    #         domain = self._map_skill_to_domain(skill_name)
    #         if score < 70:
    #             skill_profile['domain_weaknesses'][domain].append((skill_name, score))
    #         else:
    #             skill_profile['domain_strengths'][domain].append((skill_name, score))
        
    #     skill_profile['overall_performance'] = total_score / skill_count if skill_count > 0 else 0
    #     skill_profile['skill_distribution'] = self._calculate_skill_distribution(user_skills)
        
    #     return skill_profile

    def _analyze_assessment(self, assessment_data):
        """Enhanced assessment analysis with skill profiling"""
        # The new format: assessment_data[0]['assessment_result']['skills'][0]['sub_skills']
        user_skills = []
        try:
            for skill_group in assessment_data[0]['assessment_result']['skills']:
                user_skills.extend(skill_group['sub_skills'])
        except Exception as e:
            logger.error(f"Unexpected assessment JSON structure: {e}")
            raise

        skill_profile = {
            'weak_skills': [],
            'strong_skills': [],
            'skill_distribution': {},
            'overall_performance': 0,
            'domain_strengths': defaultdict(list),
            'domain_weaknesses': defaultdict(list)
        }

        total_score = 0
        skill_count = 0

        for skill in user_skills:
            score = skill['score']
            skill_name = skill['sub_skill_code'].lower().strip()
            weightage = skill.get('Weightage', 1)

            total_score += score
            skill_count += 1

            # Categorize skills
            if score < 60:
                skill_profile['weak_skills'].append({
                    'name': skill_name,
                    'score': score,
                    'weightage': weightage,
                    'priority': 'high' if score < 40 else 'medium',
                    'domain': self._map_skill_to_domain(skill_name)
                })
            elif score >= 80:
                skill_profile['strong_skills'].append({
                    'name': skill_name,
                    'score': score,
                    'domain': self._map_skill_to_domain(skill_name)
                })

            # Domain analysis
            domain = self._map_skill_to_domain(skill_name)
            if score < 70:
                skill_profile['domain_weaknesses'][domain].append((skill_name, score))
            else:
                skill_profile['domain_strengths'][domain].append((skill_name, score))

        skill_profile['overall_performance'] = total_score / skill_count if skill_count > 0 else 0
        skill_profile['skill_distribution'] = self._calculate_skill_distribution(user_skills)

        return skill_profile

    
    def _map_skill_to_domain(self, skill_name):
        """Map individual skills to broader domains using hierarchy"""
        skill_lower = skill_name.lower()
        
        for domain, keywords in self.skill_hierarchy.items():
            if any(keyword in skill_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _calculate_skill_distribution(self, skills):
        """Calculate distribution of skills across different performance levels"""
        distribution = {'poor': 0, 'average': 0, 'good': 0, 'excellent': 0}
        
        for skill in skills:
            score = skill['score']
            if score < 50:
                distribution['poor'] += 1
            elif score < 70:
                distribution['average'] += 1
            elif score < 85:
                distribution['good'] += 1
            else:
                distribution['excellent'] += 1
        
        return distribution
    
    def recommend_courses_hybrid(self, skill_profile, top_n=5):
        """
        Hybrid recommendation approach combining multiple strategies:
        1. Content-based filtering (skill matching)
        2. Difficulty progression
        3. Domain balancing
        4. Learning path optimization
        """
        try:
            recommendations = []
            
            # Strategy 1: Direct skill gap addressing (40% weight)
            skill_based_recs = self._get_skill_based_recommendations(skill_profile, weight=0.4)
            
            # Strategy 2: Domain strengthening (30% weight)
            domain_based_recs = self._get_domain_based_recommendations(skill_profile, weight=0.3)
            
            # Strategy 3: Progressive learning (20% weight)
            progressive_recs = self._get_progressive_recommendations(skill_profile, weight=0.2)
            
            # Strategy 4: Complementary skills (10% weight)
            complementary_recs = self._get_complementary_recommendations(skill_profile, weight=0.1)
            
            # Combine all recommendations
            all_recommendations = {}
            
            for rec_list, weight in [(skill_based_recs, 0.4), (domain_based_recs, 0.3), 
                                   (progressive_recs, 0.2), (complementary_recs, 0.1)]:
                for course_id, score in rec_list:
                    if course_id not in all_recommendations:
                        all_recommendations[course_id] = 0
                    all_recommendations[course_id] += score * weight
            
            # Sort and get top N recommendations
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            top_course_ids = [course_id for course_id, score in sorted_recs[:top_n]]
            
            # Create detailed recommendation DataFrame
            recommendations_df = self._create_recommendation_details(top_course_ids, all_recommendations, skill_profile)
            
            return recommendations_df
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            raise
    
    def _get_skill_based_recommendations(self, skill_profile, weight=1.0):
        """Get recommendations based on weak skills"""
        recommendations = []
        
        for weak_skill in skill_profile['weak_skills']:
            skill_name = weak_skill['name']
            priority_multiplier = 2.0 if weak_skill['priority'] == 'high' else 1.5
            
            if self.use_semantic_embeddings:
                similarities = self._calculate_semantic_similarity([skill_name])
            else:
                similarities = self._calculate_tfidf_similarity([skill_name])
            
            for idx, similarity in enumerate(similarities):
                course_id = self.course_df.iloc[idx]['Course ID'] # type: ignore
                score = similarity * priority_multiplier * weak_skill['weightage']
                recommendations.append((course_id, score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_domain_based_recommendations(self, skill_profile, weight=1.0):
        """Get recommendations to strengthen weak domains"""
        recommendations = []
        
        for domain, weak_skills in skill_profile['domain_weaknesses'].items():
            if domain == 'general':
                continue
                
            domain_courses = self.course_df[ # type: ignore
                self.course_df['Domain'].str.lower().str.contains(domain, na=False) | # type: ignore
                self.course_df['enhanced_text'].str.contains(domain, na=False) # type: ignore
            ]
            
            avg_weakness = np.mean([score for _, score in weak_skills])
            domain_priority = (100 - avg_weakness) / 100  # Higher priority for weaker domains
            
            for _, course in domain_courses.iterrows():
                score = domain_priority * 0.8  # Base score for domain match
                recommendations.append((course['Course ID'], score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_progressive_recommendations(self, skill_profile, weight=1.0):
        """Get recommendations based on learning progression"""
        recommendations = []
        user_level = self._estimate_user_level(skill_profile)
        
        for _, course in self.course_df.iterrows(): # type: ignore
            course_difficulty = course['difficulty']
            
            # Prefer courses slightly above user's current level
            if course_difficulty == user_level + 1:
                score = 0.9
            elif course_difficulty == user_level:
                score = 0.7
            elif course_difficulty == user_level + 2:
                score = 0.5
            else:
                score = 0.2
            
            recommendations.append((course['Course ID'], score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_complementary_recommendations(self, skill_profile, weight=1.0):
        """Get recommendations for complementary skills"""
        recommendations = []
        strong_domains = set(skill_profile['domain_strengths'].keys())
        
        # Complementary skill mapping
        complementary_map = {
            'programming': ['data_science', 'web_development'],
            'data_science': ['programming', 'cloud'],
            'web_development': ['programming', 'mobile'],
            'iot': ['programming', 'cloud'],
            'cloud': ['programming', 'cybersecurity']
        }
        
        for strong_domain in strong_domains:
            if strong_domain in complementary_map:
                for comp_domain in complementary_map[strong_domain]:
                    comp_courses = self.course_df[ # type: ignore
                        self.course_df['enhanced_text'].str.contains(comp_domain, na=False) # type: ignore
                    ]
                    
                    for _, course in comp_courses.iterrows():
                        score = 0.6  # Moderate score for complementary skills
                        recommendations.append((course['Course ID'], score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_semantic_similarity(self, query_skills):
        """Calculate semantic similarity using sentence transformers"""
        if not self.use_semantic_embeddings or not self.model:
            return self._calculate_tfidf_similarity(query_skills)
        
        query_text = ' '.join(query_skills)
        query_embedding = self.model.encode([query_text])
        
        course_embeddings = np.array([emb for emb in self.course_df['embeddings']]) # type: ignore
        similarities = cosine_similarity(query_embedding, course_embeddings)[0]
        
        return similarities
    
    def _calculate_tfidf_similarity(self, query_skills):
        """Fallback TF-IDF similarity calculation"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        course_texts = self.course_df['enhanced_text'].tolist() # type: ignore
        tfidf_matrix = vectorizer.fit_transform(course_texts)
        
        query_text = ' '.join(query_skills)
        query_vector = vectorizer.transform([query_text])
        
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        return similarities
    
    def _estimate_user_level(self, skill_profile):
        """Estimate user's current skill level"""
        overall_performance = skill_profile['overall_performance']
        
        if overall_performance < 50:
            return 1  # Beginner
        elif overall_performance < 75:
            return 2  # Intermediate
        else:
            return 3  # Advanced
    
    def _create_recommendation_details(self, course_ids, scores, skill_profile):
        """Create detailed recommendation DataFrame with explanations"""
        recommendations = []
        
        for course_id in course_ids:
            course_row = self.course_df[self.course_df['Course ID'] == course_id].iloc[0] # type: ignore
            
            recommendation = {
                'Course ID': course_id,
                'Course Title': course_row['Course Title'],
                'Domain': course_row['Domain'],
                'Category': course_row.get('Category', ''),
                'Difficulty': self._get_difficulty_label(course_row['difficulty']),
                'Relevance Score': round(scores[course_id], 3),
                'Recommendation Reason': self._generate_recommendation_reason(course_row, skill_profile),
                'Expected Impact': self._calculate_expected_impact(course_row, skill_profile)
            }
            
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)
    
    def _get_difficulty_label(self, difficulty_level):
        """Convert difficulty number to label"""
        labels = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}
        return labels.get(difficulty_level, 'Intermediate')
    
    def _generate_recommendation_reason(self, course_row, skill_profile):
        """Generate human-readable recommendation reason"""
        reasons = []
        course_domain = self._map_skill_to_domain(course_row['enhanced_text'])
        
        # Check if it addresses weak skills
        weak_skill_names = [skill['name'] for skill in skill_profile['weak_skills']]
        if any(skill in course_row['enhanced_text'].lower() for skill in weak_skill_names):
            reasons.append("Addresses your weak skills")
        
        # Check if it strengthens weak domains
        if course_domain in skill_profile['domain_weaknesses']:
            reasons.append(f"Strengthens your {course_domain} domain")
        
        # Check if it's complementary
        strong_domains = set(skill_profile['domain_strengths'].keys())
        if course_domain not in strong_domains and len(strong_domains) > 0:
            reasons.append("Complements your existing strengths")
        
        return "; ".join(reasons) if reasons else "General skill enhancement"
    
    def _calculate_expected_impact(self, course_row, skill_profile):
        """Calculate expected learning impact"""
        impact_score = 0
        course_domain = self._map_skill_to_domain(course_row['enhanced_text'])
        
        # Higher impact if addressing weak domains
        if course_domain in skill_profile['domain_weaknesses']:
            avg_weakness = np.mean([score for _, score in skill_profile['domain_weaknesses'][course_domain]])
            impact_score += (100 - avg_weakness) / 20  # Scale to 0-5
        
        # Moderate impact if complementary
        elif course_domain not in skill_profile['domain_strengths']:
            impact_score += 3
        
        return min(round(impact_score, 1), 5.0) # type: ignore

# Usage Example
def main():
    # Initialize the enhanced recommender
    recommender = EnhancedCourseRecommender(use_semantic_embeddings=True)
    
    # File paths
    COURSE_FILE = "/home/artisans15/projects/course_recommendation/data/courses.ods"
    ASSESSMENT_FILE = "/home/artisans15/projects/course_recommendation/data/assessments/user_assessment.json"
    
    try:
        # Load data
        course_df = recommender.load_course_data(COURSE_FILE)
        skill_profile = recommender.load_assessment_data(ASSESSMENT_FILE)
        
        # Generate recommendations
        recommendations = recommender.recommend_courses_hybrid(skill_profile, top_n=5)
        
        # Display results
        print("\n" + "="*80)
        print("ENHANCED COURSE RECOMMENDATION SYSTEM")
        print("="*80)
        
        print(f"\nUser Skill Profile Summary:")
        print(f"Overall Performance: {skill_profile['overall_performance']:.1f}%")
        print(f"Weak Skills Count: {len(skill_profile['weak_skills'])}")
        print(f"Strong Skills Count: {len(skill_profile['strong_skills'])}")
        
        print(f"\nTop Weak Skills:")
        for skill in skill_profile['weak_skills'][:3]:
            print(f"  - {skill['name']}: {skill['score']}% ({skill['priority']} priority)")
        
        print(f"\nTop 5 Course Recommendations:")
        print("-" * 80)
        for idx, row in recommendations.iterrows():
            print(f"{idx+1}. {row['Course Title']}") # type: ignore
            # print(f"   Domain: {row['Domain']} | Difficulty: {row['Difficulty']}")
            # print(f"   Relevance: {row['Relevance Score']:.3f} | Impact: {row['Expected Impact']}/5")
            print(f"   Reason: {row['Recommendation Reason']}")
            print()
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return None

if __name__ == "__main__":
    main()





# import pandas as pd
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# import logging
# from pathlib import Path

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Step 1A: Load and Preprocess Course Data
# def load_course_data(file_path):
#     """
#     Load course data from an Excel file.
#     Args:
#         file_path (str): File path to an Excel file containing course data.
#     Returns:
#         pandas.DataFrame: DataFrame containing the course data, with columns 'Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and an additional column 'text' combining Course Title and Domain.
#     Raises:
#         ValueError: If the file format is not supported or if the expected columns are not found in the file.
#     """
#     try:
#         file_ext = Path(file_path).suffix.lower()
#         if file_ext == '.xlsx':
#             course_df = pd.read_excel(file_path, engine='openpyxl')
#         elif file_ext == '.ods':
#             course_df = pd.read_excel(file_path, engine='odf')
#         else:
#             raise ValueError(f"Unsupported file format: {file_ext}. Use .xlsx or .ods.")
        
#         logger.info("Available columns in course data: %s", course_df.columns.tolist())
        
#         expected_columns = ['Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?']
#         if not all(col in course_df.columns for col in expected_columns):
#             raise ValueError(f"Expected columns not found. Expected: {', '.join(expected_columns)}. Found: {course_df.columns.tolist()}")
        
#         # Combine Course Title and Domain for text processing
#         course_df['text'] = course_df.apply(
#             lambda row: f"{row['Course Title'].lower()} {row['Domain'].lower().replace(' ', '-')}"*3, axis=1)
        
#         # Log IoT-related courses for debugging
#         iot_courses = course_df[course_df['Domain'].str.lower() == 'iot']
#         logger.info("IoT courses in dataset: %s", iot_courses[['Course ID', 'Course Title', 'Domain']].to_dict())
        
#         return course_df
#     except Exception as e:
#         logger.error("Error loading course data: %s", str(e))
#         raise

# # Step 1B: Load and Parse User Assessment
# def load_assessment_data(file_path):
#     """
#     Load the user's assessment data from a JSON file.
#     Parameters
#     ----------
#     file_path : str
#         Path to the JSON file containing the assessment data.
#     Returns
#     -------
#     data : dict
#         The loaded assessment data.
#     Raises
#     ------
#     Exception
#         If there is an error loading the file.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             return json.load(file)
#     except Exception as e:
#         logger.error("Error loading assessment data: %s", str(e))
#         raise

# # Step 2: Identify User's Skill Gaps with Dynamic Domain Mapping
# def identify_skill_gaps(assessment_data, course_df, benchmark_score=70, num_clusters=10):
#     """
#     Identify user's skill gaps by comparing their assessment scores against a benchmark score (default: 70).
#     Dynamically map skills to domains using K-Means clustering based on Course Title and Domain.
#     Parameters
#     ----------
#     assessment_data : dict
#         The loaded assessment data from a JSON file.
#     course_df : pandas.DataFrame
#         DataFrame containing course data to extract domains for mapping.
#     benchmark_score : int, optional
#         The minimum score to consider a skill as a gap. Defaults to 70.
#     num_clusters : int, optional
#         Number of clusters for K-Means to group skills into domains. Defaults to 10.
#     Returns
#     -------
#     weak_skills : list
#         A list of dictionaries containing the identified skill gaps with their weightage and dynamically assigned domain.
#     Raises
#     ------
#     Exception
#         If there is an error identifying the skill gaps or clustering.
#     """
#     try:
#         user_sub_skills = assessment_data[0]['assessment_result']['skills'][0]['sub_skills']
#         weak_skills = []
        
#         # Extract domains from course data
#         course_domains = course_df['Domain'].str.lower().unique().tolist()
        
#         # Extract all unique skills from assessment
#         all_skills = [s['sub_skill_code'].lower().strip() for s in user_sub_skills]
#         if not all_skills:
#             raise ValueError("No skills found for clustering.")
        
#         # Create skill texts for clustering, using skill names
#         skill_texts = [f"{skill} {'iot' if 'iot' in skill else 'general'}"*3 for skill in all_skills]
        
#         # Convert skills to TF-IDF vectors
#         tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
#         skill_vectors = tfidf_vectorizer.fit_transform(skill_texts)
#         logger.info("TF-IDF features for skill mapping: %s", tfidf_vectorizer.get_feature_names_out().tolist())
        
#         # Apply K-Means clustering
#         num_clusters = min(num_clusters, len(all_skills))
#         kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#         cluster_labels = kmeans.fit_predict(skill_vectors)
        
#         # Map clusters to course domains where possible
#         skill_domain_mapping = {}
#         for skill, label in zip(all_skills, cluster_labels):
#             # Assign 'iot' domain if skill contains 'iot' and 'iot' is in course domains
#             if 'iot' in skill and 'iot' in course_domains:
#                 skill_domain_mapping[skill] = 'iot'
#             else:
#                 skill_domain_mapping[skill] = f"Domain_{label}"
        
#         # Assign domains to weak skills
#         for sub_skill in user_sub_skills:
#             if sub_skill['score'] < benchmark_score:
#                 skill_code = sub_skill['sub_skill_code'].lower().strip()
#                 skill_data = {
#                     'sub_skill_code': skill_code,
#                     'weightage': sub_skill.get('Weightage', 1),
#                     'domain': skill_domain_mapping.get(skill_code, 'general')
#                 }
#                 weak_skills.append(skill_data)
        
#         logger.info("Identified skill gaps: %s", [s['sub_skill_code'] for s in weak_skills])
#         logger.info("Dynamic domain mapping: %s", skill_domain_mapping)
#         return weak_skills
#     except Exception as e:
#         logger.error("Error identifying skill gaps: %s", str(e))
#         raise

# # Step 3 & 4: Enhanced Recommendation Engine with Domain Consideration
# def recommend_courses(course_df, weak_skills, top_n=5, similarity_threshold=0.1):
#     """
#     Generate course recommendations based on the user's skill gaps using Course Title and Domain.
#     Parameters
#     ----------
#     course_df : pandas.DataFrame
#         DataFrame containing the course data, with columns 'Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and 'text' combining Course Title and Domain.
#     weak_skills : list
#         A list of dictionaries containing the identified skill gaps with their weightage and domain.
#     top_n : int, optional
#         The number of top recommendations to return. Defaults to 5.
#     similarity_threshold : float, optional
#         The minimum cosine similarity score to consider a course as a recommendation. Defaults to 0.1.
#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame containing the recommended courses, with columns 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and an additional column 'relevance_score'.
#     Raises
#     ------
#     Exception
#         If there is an error generating the recommendations.
#     """
#     try:
#         # Filter courses by domains of weak skills
#         weak_domains = set(skill['domain'].lower() for skill in weak_skills if skill['domain'] != 'general')
#         if weak_domains:
#             domain_matched_courses = course_df[course_df['Domain'].str.lower().isin(weak_domains)]
#             logger.info("Courses in matching domains: %s", domain_matched_courses['Course ID'].tolist())
#             if not domain_matched_courses.empty:
#                 course_df = domain_matched_courses
        
#         tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
#         course_tfidf_matrix = tfidf_vectorizer.fit_transform(course_df['text'])
#         logger.info("TF-IDF features for courses: %s", tfidf_vectorizer.get_feature_names_out().tolist())
        
#         # Create user text from weak skills, emphasizing weightage and domain
#         user_weak_skills_text = ' '.join(f"{skill['sub_skill_code']} {skill['weightage']} {skill['domain']}"*3 
#                                        for skill in weak_skills)
#         user_tfidf_vector = tfidf_vectorizer.transform([user_weak_skills_text])
        
#         cosine_similarities = cosine_similarity(user_tfidf_vector, course_tfidf_matrix)
#         similarity_scores = cosine_similarities[0]
        
#         # Boost relevance score based on domain match
#         domain_boost = 0.3
#         for idx, row in course_df.iterrows():
#             course_domain = row['Domain'].lower()
#             for skill in weak_skills:
#                 if course_domain == skill['domain'].lower():
#                     similarity_scores[idx] += domain_boost
        
#         course_df['relevance_score'] = similarity_scores
#         filtered_courses = course_df[course_df['relevance_score'] >= similarity_threshold]
#         recommended_courses = filtered_courses.sort_values(by='relevance_score', ascending=False)
#         final_recommendations = recommended_courses.head(top_n)
        
#         logger.info("Recommended courses: %s", final_recommendations['Course ID'].tolist())
#         return final_recommendations
#     except Exception as e:
#         logger.error("Error generating recommendations: %s", str(e))
#         raise

# # Main execution
# if __name__ == "__main__":
#     # File paths
#     COURSE_FILE = "/home/artisans15/projects/course_recommendation/data/courses.ods"
#     ASSESSMENT_FILE = "/home/artisans15/projects/course_recommendation/data/assessments/user_assessment.json"
    
#     # Load and process data
#     try:
#         course_df = load_course_data(COURSE_FILE)
#         assessment_data = load_assessment_data(ASSESSMENT_FILE)
        
#         # Identify skill gaps with dynamic domain mapping
#         weak_skills = identify_skill_gaps(assessment_data, course_df, num_clusters=10)
        
#         # Generate recommendations with configurable parameters
#         recommendations = recommend_courses(course_df, weak_skills, top_n=5, similarity_threshold=0.1)
        
#         print("\nUser's Identified Skill Gaps:")
#         for skill in weak_skills:
#             print(f"- {skill['sub_skill_code']} (Weightage: {skill['weightage']}, Domain: {skill['domain']})")
        
#         print("\nTop Course Recommendations:")
#         print(recommendations[['Course ID', 'Course Title', 'Domain', 'relevance_score']])
#     except Exception as e:
#         print(f"Error: {e}")

#######################################################################################################################################################################

# import pandas as pd
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# from pathlib import Path

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Step 1A: Load and Preprocess Course Data
# def load_course_data(file_path):
#     """
#     Load course data from an Excel file.

#     Args:
#         file_path (str): File path to an Excel file containing course data.

#     Returns:
#         pandas.DataFrame: DataFrame containing the course data, with columns 'Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and two additional columns 'processed_skills' and 'skills_text' containing the cleaned and joined skills, respectively.

#     Raises:
#         ValueError: If the file format is not supported or if the expected columns are not found in the file.
#     """
#     try:
#         file_ext = Path(file_path).suffix.lower()
#         if file_ext == '.xlsx':
#             course_df = pd.read_excel(file_path, engine='openpyxl')
#         elif file_ext == '.ods':
#             course_df = pd.read_excel(file_path, engine='odf')
#         else:
#             raise ValueError(f"Unsupported file format: {file_ext}. Use .xlsx or .ods.")
        
#         logger.info("Available columns in course data: %s", course_df.columns.tolist())
        
#         expected_columns = ['Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?']
#         if not all(col in course_df.columns for col in expected_columns):
#             raise ValueError(f"Expected columns not found. Expected: {', '.join(expected_columns)}. Found: {course_df.columns.tolist()}")
        
#         skill_column = next((col for col in course_df.columns if 'skill' in col.lower()), None)
#         if skill_column is None:
#             raise ValueError("No column containing 'skill' found. Please ensure the 'Skill Areas' column exists.")
        
#         def clean_skills(skill_string):
#             if isinstance(skill_string, str):
#                 skills = skill_string.lower().replace('&', ' ').split()
#                 return [skill.strip() for skill in skills]
#             return []
        
#         course_df['processed_skills'] = course_df[skill_column].apply(clean_skills)
#         course_df['skills_text'] = course_df['processed_skills'].apply(lambda skills: ' '.join(skills))
#         return course_df
#     except Exception as e:
#         logger.error("Error loading course data: %s", str(e))
#         raise

# # Step 1B: Load and Parse User Assessment
# def load_assessment_data(file_path):
#     """
#     Load the user's assessment data from a JSON file.

#     Parameters
#     ----------
#     file_path : str
#         Path to the JSON file containing the assessment data.

#     Returns
#     -------
#     data : dict
#         The loaded assessment data.

#     Raises
#     ------
#     Exception
#         If there is an error loading the file.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             return json.load(file)
#     except Exception as e:
#         logger.error("Error loading assessment data: %s", str(e))
#         raise

# # Step 2: Identify User's Skill Gaps with Weightage and Domain Mapping
# def identify_skill_gaps(assessment_data, benchmark_score=70):
#     """
#     Identify user's skill gaps by comparing their assessment scores against a benchmark score (default: 70).

#     Parameters
#     ----------
#     assessment_data : dict
#         The loaded assessment data from a JSON file.
#     benchmark_score : int, optional
#         The minimum score to consider a skill as a gap. Defaults to 70.

#     Returns
#     -------
#     weak_skills : list
#         A list of dictionaries containing the identified skill gaps with their weightage and domain.

#     Raises
#     ------
#     Exception
#         If there is an error identifying the skill gaps.
#     """
#     try:
#         user_sub_skills = assessment_data[0]['assessment_result']['skills'][0]['sub_skills']
#         weak_skills = []
#         # Map sub-skills to potential domains (custom mapping based on skill names)
#         skill_domain_mapping = {
#             'iot-basics': 'General',
#             'iot-connectivity-basics': 'Network',
#             'iot-data-management-and-security-basics': 'Security',
#             'iot-hardware-concepts': 'General',
#             'iot-networking-and-development-basics': 'Network',
#             'iot-networking-and-development-for-industry': 'Network',
#             'python-for-iot': 'Data Analytics'
#         }
#         for sub_skill in user_sub_skills:
#             if sub_skill['score'] < benchmark_score:
#                 skill_data = {
#                     'sub_skill_code': sub_skill['sub_skill_code'].lower().strip(),
#                     'weightage': sub_skill.get('Weightage', 1),
#                     'domain': skill_domain_mapping.get(sub_skill['sub_skill_code'].lower().strip(), 'General')
#                 }
#                 weak_skills.append(skill_data)
#         logger.info("Identified skill gaps: %s", [s['sub_skill_code'] for s in weak_skills])
#         return weak_skills
#     except Exception as e:
#         logger.error("Error identifying skill gaps: %s", str(e))
#         raise

# # Step 3 & 4: Enhanced Recommendation Engine with Domain Consideration
# def recommend_courses(course_df, weak_skills, top_n=5, similarity_threshold=0.1):
#     """
#     Generate course recommendations based on the user's skill gaps with domain consideration.

#     Parameters
#     ----------
#     course_df : pandas.DataFrame
#         DataFrame containing the course data, with columns 'Sr No.', 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and two additional columns 'processed_skills' and 'skills_text' containing the cleaned and joined skills, respectively.
#     weak_skills : list
#         A list of dictionaries containing the identified skill gaps with their weightage and domain.
#     top_n : int, optional
#         The number of top recommendations to return. Defaults to 5.
#     similarity_threshold : float, optional
#         The minimum cosine similarity score to consider a course as a recommendation. Defaults to 0.1.

#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame containing the recommended courses, with columns 'Course ID', 'Course Title', 'Skill Areas', 'Domain', 'Category', 'Is Published?' and an additional column 'relevance_score' containing the relevance score of each course.

#     Raises0
#     ------
#     Exception
#         If there is an error generating the recommendations.
#     """
#     try:
#         tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
#         course_tfidf_matrix = tfidf_vectorizer.fit_transform(course_df['skills_text'])
        
#         # Weight skills based on weightage and include domain influence
#         user_weak_skills_text = ' '.join(f"{skill['sub_skill_code']} {skill['weightage']} {skill['domain']}" 
#                                        for skill in weak_skills)
#         user_tfidf_vector = tfidf_vectorizer.transform([user_weak_skills_text])
        
#         cosine_similarities = cosine_similarity(user_tfidf_vector, course_tfidf_matrix)
#         similarity_scores = cosine_similarities[0]
        
#         # Boost relevance score based on domain match
#         domain_boost = 0.2  # Adjustable boost factor
#         for idx, row in course_df.iterrows():
#             course_domain = row['Domain'].lower()
#             for skill in weak_skills:
#                 if course_domain == skill['domain'].lower():
#                     similarity_scores[idx] += domain_boost
        
#         course_df['relevance_score'] = similarity_scores
#         filtered_courses = course_df[course_df['relevance_score'] >= similarity_threshold]
#         recommended_courses = filtered_courses.sort_values(by='relevance_score', ascending=False)
#         final_recommendations = recommended_courses.head(top_n)
        
#         logger.info("Recommended courses: %s", final_recommendations['Course ID'].tolist())
#         return final_recommendations
#     except Exception as e:
#         logger.error("Error generating recommendations: %s", str(e))
#         raise

# # Main execution
# if __name__ == "__main__":
#     # File paths
#     COURSE_FILE = "/home/artisans15/projects/course_recommendation/data/courses.ods"
#     ASSESSMENT_FILE = "/home/artisans15/projects/course_recommendation/data/assessments/user_assessment.json"
    
#     # Load and process data
#     try:
#         course_df = load_course_data(COURSE_FILE)
#         assessment_data = load_assessment_data(ASSESSMENT_FILE)
        
#         # Identify skill gaps with weightage and domain
#         weak_skills = identify_skill_gaps(assessment_data)
        
#         # Generate recommendations with configurable parameters
#         recommendations = recommend_courses(course_df, weak_skills, top_n=5, similarity_threshold=0.1)
        
#         # print("\nUser's Identified Skill Gaps:")
#         # for skill in weak_skills:
#         #     print(f"- {skill['sub_skill_code']} (Weightage: {skill['weightage']}, Domain: {skill['domain']})")
        
#         print("\nTop Course Recommendations:")
#         print(recommendations[['Course ID', 'Course Title', 'Skill Areas', 'Domain', 'relevance_score']])
#     except Exception as e:
#         print(f"Error: {e}")
