from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from google import genai

app = Flask(__name__)
CORS(app)

# 1. Spacy NLP Model Engine load gochuu
nlp = spacy.load("en_core_web_sm")

# 2. Gemini Client Qopheessuu
# API Key keenya Render Environment Variables keessaa dubbisa
gemini_key = os.environ.get("GEMINI_API_KEY", "YOUR_FALLBACK_KEY_HERE")
client = genai.Client(api_key=gemini_key)

def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(cleaned_tokens)

@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    if 'file' not in request.files or 'jobs' not in request.form:
        return jsonify({"error": "File ykn Data hojii dhabameera"}), 400
        
    file = request.files['file']
    job_listings = request.form['jobs']
    
    pdf_reader = PyPDF2.PdfReader(file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
        
    cleaned_resume = preprocess_text(resume_text)
    jobs = json.loads(job_listings)
    
    recommendations = []
    for job in jobs:
        cleaned_job = preprocess_text(job['skills_required'])
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        match_percentage = round(score * 100, 2)
        recommendations.append({
            "job_id": job['id'],
            "title": job['title'],
            "match_score": f"{match_percentage}%"
        })
        
    recommendations = sorted(recommendations, key=lambda x: float(x['match_score'].replace('%','')), reverse=True)
    return jsonify({"recommendations": recommendations})

@app.route('/generate-course', methods=['POST'])
def generate_course():
    data = request.json
    target_job = data.get('target_job', 'Video Editing')
    current_skills = data.get('current_skills', 'None')

    # YouTube Tutorials links built-in dhiyeessuuf
    video_database = {
        "video editing": "https://youtube.com",
        "graphic design": "https://youtube.com",
        "coding": "https://youtube.com",
        "content creator": "https://youtube.com"
    }
    matched_video = video_database.get(target_job.lower(), "https://youtube.com")

    # Gemini'f ulaagaa (Prompt) itti fiduu akka inni JSON sirrii deebisuuf
    system_prompt = (
        f"You are an expert AI Career Coach. Create a highly personalized 4-week upskilling curriculum "
        f"for a user who wants to become a {target_job}. Their current skills are: {current_skills}. "
        f"Respond STRICTLY in a clean JSON format with these exact keys: "
        f"'career_advice' (a short text block) and 'weekly_modules' (an array of 4 objects, each containing "
        f"'week', 'topic', and 'project'). Do not wrap inside markdown blocks like ```json."
    )

    try:
        # Gemini 1.5 Flash Waamuu
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=system_prompt,
        )
        
        # Deebii Gemini irraa dhufe gara Python Dictionary tti jijjiiruu
        ai_data = json.loads(response.text.strip())
        ai_data["tutorial_video"] = matched_video
        ai_data["job_title"] = target_job
        
        return jsonify(ai_data)
        
    except Exception as e:
        # Gemini yoo sababa API Key tiin hojjechuu baate, fallback template salphaa deebisa
        return jsonify({
            "job_title": target_job,
            "tutorial_video": matched_video,
            "weekly_modules": [
                {"week": "Week 1", "topic": f"Basics of {target_job}", "project": "Complete initial practice tasks."},
                {"week": "Week 2", "topic": "Intermediate tools workflow", "project": "Create your first mini portfolio piece."},
                {"week": "Week 3", "topic": "Advanced systems integration", "project": "Build a simulation project."},
                {"week": "Week 4", "topic": "Portfolio optimization", "project": "Apply to listings via this dashboard."}
            ],
            "career_advice": "Focus heavily on practical daily projects to fast-track your onboarding experience."
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
