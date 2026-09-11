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
CORS(app, resources={r"/*": {"origins": "*"}})

nlp = spacy.load("en_core_web_sm")

gemini_key = os.environ.get("GEMINI_API_KEY", "YOUR_FALLBACK_KEY_HERE")
client = genai.Client(api_key=gemini_key)

def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(cleaned_tokens)

@app.route('/')
def home():
    return jsonify({"status": "healthy", "message": "Nile Job AI Backend is running perfectly!"})

@app.route('/recommend', methods=['POST'])
def recommend_jobs():
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-course', methods=['POST'])
def generate_course():
    if request.is_json:
        data = request.json
    else:
        data = request.form
        
    target_job = data.get('target_job', 'Coding') if data else 'Coding'
    current_skills = data.get('current_skills', 'None') if data else 'None'

    video_database = {
        "video editing": "https://youtube.com",
        "graphic design": "https://youtube.com",
        "coding": "https://youtube.com",
        "content creator": "https://youtube.com"
    }
    matched_video = video_database.get(target_job.lower(), "https://youtube.com")

    # Gemini irraa JSON qulqulluu argachuuf prompt tajaajila addaa
    system_prompt = (
        f"You are an expert AI Career Coach. Create a highly personalized 4-week upskilling curriculum "
        f"for a user who wants to become a {target_job}. Their current skills and question are: {current_skills}. "
        f"Respond ONLY with a raw JSON object. Do not include markdown formatting like ```json or backticks. "
        f"The JSON must have exactly these keys: "
        f"'career_advice' (a tailored answer or advice based on their query) and 'weekly_modules' (an array of 4 objects, each containing "
        f"'week', 'topic', and 'project')."
    )

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=system_prompt,
        )
        
        # Markdown backticks yoo dhufe qulqulleessuu
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        ai_data = json.loads(raw_text)
        ai_data["tutorial_video"] = matched_video
        ai_data["job_title"] = target_job
        return jsonify(ai_data)
        
    except Exception as e:
        # Gemini irraa yoo sirriitti dubbisuu baate, gaaffii deebisuuf deebii dynamic ta'e qopheessuu
        return jsonify({
            "job_title": target_job,
            "tutorial_video": matched_video,
            "weekly_modules": [
                {"week": "Week 1", "topic": f"Basics of {target_job}", "project": f"Research starting kits for {target_job}."},
                {"week": "Week 2", "topic": "Core concepts and setup", "project": "Build your first introductory template."},
                {"week": "Week 3", "topic": "Intermediate project workflow", "project": "Develop a real-world simulation task."},
                {"week": "Week 4", "topic": "Portfolio optimization", "project": "Publish your portfolio and prepare for interviews."}
            ],
            "career_advice": f"You asked about '{current_skills}'. To start in {target_job}, focus on consistent practice every single day."
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
