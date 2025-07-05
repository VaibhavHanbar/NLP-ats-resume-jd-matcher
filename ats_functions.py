import re
import nltk
import docx2txt
import pdfplumber
import string
import pandas as pd
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('maxent_ne_chunker_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Set up lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Common skills database (simplified version)
SKILLS_DB = {
    'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'scala', 'r', 'golang', 'swift'],
    'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'dynamodb'],
    'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js', 'express'],
    'data_science': ['machine learning', 'data analysis', 'statistics', 'ai', 'deep learning', 'nlp', 
                    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'devops', 'ci/cd'],
    'soft_skills': ['communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
                    'time management', 'adaptability', 'creativity']
}

# Dictionary of common education levels
EDUCATION_LEVELS = {
    'high school': 1,
    'associate': 2, 
    'bachelor': 3,
    'undergraduate': 3,
    'master': 4,
    'mba': 4,
    'phd': 5,
    'doctorate': 5
}

# Function to extract text from resume
def extract_text_from_resume(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
        elif file_path.endswith('.docx'):
            text = docx2txt.process(file_path)
        else:
            raise Exception("Unsupported file format. Upload PDF or DOCX only.")
    except Exception as e:
        print(f"Error extracting text from file: {str(e)}")
        raise
    return text

# Basic preprocessing function
def preprocess_text(text):
    try:
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenization
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back to string
        clean_text = ' '.join(tokens)
        return clean_text
    except Exception as e:
        print(f"Error preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails

# Function to segment text into sections (basic version)
def segment_document(text):
    sections = {}
    
    try:
        # Define common section headers
        patterns = {
            'education': r'education|academic|qualification',
            'experience': r'experience|work history|employment|professional background',
            'skills': r'skills|technical skills|competencies|proficiencies|expertise',
            'projects': r'projects|portfolio|works',
            'contact': r'contact|personal details|phone|email',
            'summary': r'summary|profile|objective|about me'
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Find positions of all section headers
        positions = []
        for section, pattern in patterns.items():
            for match in re.finditer(r'(^|\n)(' + pattern + r')(\:|\s|\n)', text_lower):
                positions.append((match.start(), section))
        
        # Sort positions by their occurrence in the text
        positions.sort()
        
        # Extract sections based on positions
        for i in range(len(positions)):
            start_pos = positions[i][0]
            section_name = positions[i][1]
            
            # End position is either the start of the next section or the end of the text
            end_pos = positions[i+1][0] if i < len(positions)-1 else len(text)
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            sections[section_name] = section_content
        
    except Exception as e:
        print(f"Error segmenting document: {str(e)}")
    
    # If no sections were found, use the entire text as 'full_text'
    if not sections:
        sections['full_text'] = text
    
    return sections

# Function to extract years of experience
def extract_years_of_experience(text):
    try:
        # Patterns for years of experience
        patterns = [
            r'(\d+)\+?\s*(?:years|yrs)(?:\s*of)?\s*(?:experience|exp)',
            r'(?:experience|exp)(?:\s*of)?\s*(\d+)\+?\s*(?:years|yrs)',
            r'(\d+)\+?\s*(?:years|yrs)'
        ]
        
        max_years = 0
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        return max_years
    except Exception as e:
        print(f"Error extracting years of experience: {str(e)}")
        return 0

# Function to detect education level
def extract_education_level(text):
    try:
        text = text.lower()
        highest_level = 0
        found_level = None
        
        for degree, level in EDUCATION_LEVELS.items():
            if degree in text and level > highest_level:
                highest_level = level
                found_level = degree
                
        return found_level, highest_level
    except Exception as e:
        print(f"Error extracting education level: {str(e)}")
        return None, 0

# Function to extract skills from text
def extract_skills(text, custom_skills=None):
    try:
        text = text.lower()
        found_skills = {}
        
        # Combine with any custom skills provided
        skills_db = SKILLS_DB.copy()
        if custom_skills:
            skills_db['custom'] = custom_skills
        
        # Look for skills in the text
        for category, skills_list in skills_db.items():
            category_skills = []
            for skill in skills_list:
                # For multi-word skills
                if ' ' in skill:
                    if skill in text:
                        category_skills.append(skill)
                # For single-word skills, use word boundary check
                else:
                    pattern = r'\b' + re.escape(skill) + r'\b'
                    if re.search(pattern, text):
                        category_skills.append(skill)
            
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")
        return {}

# Function to count frequency of skills
def count_skills_frequency(text):
    try:
        all_skills = []
        for category, skills in SKILLS_DB.items():
            for skill in skills:
                if skill in text.lower():
                    all_skills.append(skill)
        
        return Counter(all_skills)
    except Exception as e:
        print(f"Error counting skill frequency: {str(e)}")
        return Counter()

# Function to extract named entities
def extract_entities(text):
    entities = {
        'organizations': [],
        'persons': [],
        'locations': [],
        'dates': [],
        'titles': []
    }
    
    try:
        # Extract common job titles
        job_titles = [
            'developer', 'engineer', 'manager', 'director', 'analyst', 
            'scientist', 'designer', 'administrator', 'consultant', 'specialist'
        ]
        
        for title in job_titles:
            pattern = r'\b\w+\s+' + title + r'\b|\b' + title + r'\b'
            matches = re.findall(pattern, text.lower())
            entities['titles'].extend(matches)
        
        # Use NLTK for named entity recognition
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tagged = pos_tag(words)
            named_entities = ne_chunk(tagged)
            
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity = ' '.join(c[0] for c in chunk)
                    if chunk.label() == 'ORGANIZATION':
                        entities['organizations'].append(entity)
                    elif chunk.label() == 'PERSON':
                        entities['persons'].append(entity)
                    elif chunk.label() == 'GPE' or chunk.label() == 'LOCATION':
                        entities['locations'].append(entity)
        
        # Remove duplicates and keep unique entities
        for category in entities:
            entities[category] = list(set(entities[category]))
    
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
    
    return entities

# Calculate cosine similarity between two texts
def calculate_cosine_similarity(text1, text2):
    try:
        if not text1 or not text2:
            return 0.0
            
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Check for zero vectors
        norm1 = (tfidf_matrix[0] * tfidf_matrix[0].T).toarray()[0][0]**0.5
        norm2 = (tfidf_matrix[1] * tfidf_matrix[1].T).toarray()[0][0]**0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = ((tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0] / (norm1 * norm2))
        return round(similarity * 100, 2)
    except Exception as e:
        print(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

# Match requirements with resume
def match_requirements(jd_text, resume_text):
    try:
        # Extract key phrases that might be requirements
        requirement_patterns = [
            r'required:?\s*(.*?)(?:\.|$)',
            r'requirements:?\s*(.*?)(?:\.|$)',
            r'qualifications:?\s*(.*?)(?:\.|$)',
            r'must have:?\s*(.*?)(?:\.|$)',
            r'should have:?\s*(.*?)(?:\.|$)',
            r'need to have:?\s*(.*?)(?:\.|$)',
        ]
        
        requirements = []
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            requirements.extend(matches)
        
        # If no specific requirements found, extract sentences with key requirement words
        if not requirements:
            sentences = nltk.sent_tokenize(jd_text)
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['require', 'must', 'should', 'need', 'skill']):
                    requirements.append(sentence)
        
        # Match each requirement with the resume
        requirement_matches = []
        
        for req in requirements:
            # Skip very short requirements
            if len(req.split()) < 3:
                continue
                
            similarity = calculate_cosine_similarity(req, resume_text)
            requirement_matches.append({
                'requirement': req,
                'match_percentage': similarity
            })
        
        # Sort by match percentage
        requirement_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        return requirement_matches
    except Exception as e:
        print(f"Error matching requirements: {str(e)}")
        return []

# Main analysis function
def analyze_resume_jd_match(resume_path, job_description):
    try:
        # Extract texts
        resume_text = extract_text_from_resume(resume_path)
        jd_text = job_description
        
        # Segment documents
        resume_sections = segment_document(resume_text)
        jd_sections = segment_document(jd_text)
        
        # Preprocess for general similarity
        processed_resume = preprocess_text(resume_text)
        processed_jd = preprocess_text(jd_text)
        
        # Overall similarity
        overall_similarity = calculate_cosine_similarity(processed_resume, processed_jd)
        
        # Section similarities
        section_similarities = {}
        for jd_section_name, jd_section_text in jd_sections.items():
            best_match = 0
            matched_section = None
            
            # Find best matching section in resume
            for resume_section_name, resume_section_text in resume_sections.items():
                similarity = calculate_cosine_similarity(
                    preprocess_text(jd_section_text),
                    preprocess_text(resume_section_text)
                )
                if similarity > best_match:
                    best_match = similarity
                    matched_section = resume_section_name
            
            if matched_section:
                section_similarities[jd_section_name] = {
                    'matched_with': matched_section,
                    'similarity': best_match
                }
        
        # Extract skills
        jd_skills = extract_skills(jd_text)
        resume_skills = extract_skills(resume_text)
        
        # Count skill frequencies
        skill_frequencies = count_skills_frequency(resume_text)
        
        # Calculate skills match
        skills_match = {
            'matched': {},
            'missing': {},
            'additional': {}
        }
        
        # Find matched and missing skills
        for category, skills in jd_skills.items():
            matched_skills = []
            missing_skills = []
            
            # Check if category exists in resume
            if category in resume_skills:
                for skill in skills:
                    if skill in resume_skills[category]:
                        matched_skills.append(skill)
                    else:
                        missing_skills.append(skill)
            else:
                missing_skills = skills
            
            if matched_skills:
                skills_match['matched'][category] = matched_skills
            if missing_skills:
                skills_match['missing'][category] = missing_skills
        
        # Find additional skills in resume
        for category, skills in resume_skills.items():
            additional_skills = []
            
            # Check if category exists in job description
            if category in jd_skills:
                for skill in skills:
                    if skill not in jd_skills[category]:
                        additional_skills.append(skill)
            else:
                additional_skills = skills
            
            if additional_skills:
                skills_match['additional'][category] = additional_skills
        
        # Calculate skills match percentage
        total_jd_skills = sum(len(skills) for skills in jd_skills.values())
        skills_match_percentage = 0
        if total_jd_skills > 0:
            total_matched_skills = sum(len(skills) for skills in skills_match['matched'].values())
            skills_match_percentage = (total_matched_skills / total_jd_skills * 100)
        
        # Extract experience information
        jd_experience = extract_years_of_experience(jd_text)
        resume_experience = extract_years_of_experience(resume_text)
        experience_match = resume_experience >= jd_experience
        
        # Extract education information
        jd_education, jd_edu_level = extract_education_level(jd_text)
        resume_education, resume_edu_level = extract_education_level(resume_text)
        education_match = resume_edu_level >= jd_edu_level
        
        # Extract entities
        jd_entities = extract_entities(jd_text)
        resume_entities = extract_entities(resume_text)
        
        # Requirement matching
        requirement_matches = match_requirements(jd_text, resume_text)
        
        # Calculate average requirement match
        avg_req_match = 0
        if requirement_matches:
            avg_req_match = sum(req['match_percentage'] for req in requirement_matches) / len(requirement_matches)
        
        # Weighted final score (can be adjusted based on importance)
        weights = {
            'overall_similarity': 0.2,
            'skills_match': 0.4,
            'experience_match': 0.2,
            'education_match': 0.1,
            'requirement_match': 0.1
        }
        
        final_score = (
            weights['overall_similarity'] * overall_similarity +
            weights['skills_match'] * skills_match_percentage +
            weights['experience_match'] * (100 if experience_match else 50) +
            weights['education_match'] * (100 if education_match else 50) +
            weights['requirement_match'] * avg_req_match
        )
        
        # Compile results
        results = {
            'overall_similarity': overall_similarity,
            'section_similarities': section_similarities,
            'skills_analysis': {
                'skills_match': skills_match,
                'skills_match_percentage': round(skills_match_percentage, 2),
                'skill_frequencies': dict(skill_frequencies)
            },
            'experience_analysis': {
                'jd_required_years': jd_experience,
                'resume_years': resume_experience,
                'experience_match': experience_match
            },
            'education_analysis': {
                'jd_required_education': jd_education,
                'resume_education': resume_education,
                'education_match': education_match
            },
            'entity_analysis': {
                'jd_entities': jd_entities,
                'resume_entities': resume_entities
            },
            'requirement_analysis': {
                'requirement_matches': requirement_matches,
                'average_requirement_match': round(avg_req_match, 2)
            },
            'final_score': round(final_score, 2)
        }
        
        return results
    
    except Exception as e:
        print(f"Error analyzing resume and job description: {str(e)}")
        print(traceback.format_exc())  # Print full stack trace
        return None

# Function to display results in a readable format
def display_results(results):
    if not results:
        print("No results to display.")
        return
    
    try:
        print("\n========== RESUME-JD MATCH ANALYSIS ==========")
        print(f"OVERALL MATCH SCORE: {results['final_score']}%")
        print("\n----- SIMILARITY BREAKDOWN -----")
        print(f"Overall Text Similarity: {results['overall_similarity']}%")
        print(f"Skills Match: {results['skills_analysis']['skills_match_percentage']}%")
        
        print("\n----- SKILLS ANALYSIS -----")
        if results['skills_analysis']['skills_match']['matched']:
            print("MATCHED SKILLS:")
            for category, skills in results['skills_analysis']['skills_match']['matched'].items():
                print(f"  {category.upper()}: {', '.join(skills)}")
        
        if results['skills_analysis']['skills_match']['missing']:
            print("\nMISSING SKILLS (in JD but not in Resume):")
            for category, skills in results['skills_analysis']['skills_match']['missing'].items():
                print(f"  {category.upper()}: {', '.join(skills)}")
        
        if results['skills_analysis']['skills_match']['additional']:
            print("\nADDITIONAL SKILLS (in Resume but not in JD):")
            for category, skills in results['skills_analysis']['skills_match']['additional'].items():
                print(f"  {category.upper()}: {', '.join(skills)}")
        
        print("\n----- EXPERIENCE ANALYSIS -----")
        print(f"JD Required Years: {results['experience_analysis']['jd_required_years']}")
        print(f"Resume Years: {results['experience_analysis']['resume_years']}")
        print(f"Experience Match: {'YES' if results['experience_analysis']['experience_match'] else 'NO'}")
        
        print("\n----- EDUCATION ANALYSIS -----")
        print(f"JD Required Education: {results['education_analysis']['jd_required_education'] or 'Not specified'}")
        print(f"Resume Education: {results['education_analysis']['resume_education'] or 'Not detected'}")
        print(f"Education Match: {'YES' if results['education_analysis']['education_match'] else 'NO'}")
        
        print("\n----- TOP REQUIREMENT MATCHES -----")
        for i, req in enumerate(results['requirement_analysis']['requirement_matches'][:5], 1):
            print(f"{i}. Requirement: {req['requirement']}")
            print(f"   Match: {req['match_percentage']}%")
        
        print(f"\nAverage Requirement Match: {results['requirement_analysis']['average_requirement_match']}%")
        print("=============================================")
    except Exception as e:
        print(f"Error displaying results: {str(e)}")

# Main function
def main(resume_path, job_description):
    try:
        results = analyze_resume_jd_match(resume_path, job_description)
        if results:
            display_results(results)
            return results
        else:
            print("Analysis failed.")
            return None
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        print(traceback.format_exc())  # Print full stack trace
        return None

# Example usage (commented out)
if __name__ == "__main__":
    resume_path = r"C:\Users\Lenovo\Desktop\Desktop\Resumes\CV_Vaibhav_Hanbar_Data_Fresher_NLP.pdf"
    job_description = """
    Location : Gurugram Headquarter
    """
    results = main(resume_path, job_description)

