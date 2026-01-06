import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
import PyPDF2

# ==============================
# ENVIRONMENT SETUP
# ==============================
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file.")

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = genai.Client(api_key=api_key)

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ==============================
# RESUME PARSER (LLM)
# ==============================
def parse_resume(resume_text):
    prompt = f"""
You are a resume parser.
Extract:
- Skills
- Experience summary
- Education
- Tools & technologies
Resume:
{resume_text}
Return in bullet points.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ==============================
# JOB DESCRIPTION PARSER
# ==============================
def parse_job_description(jd_text):
    prompt = f"""
Extract:
- Required skills
- Responsibilities
- Preferred qualifications
Job Description:
{jd_text}
Return in bullet points.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ==============================
# ATS MATCHING
# ==============================
def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
You are an Applicant Tracking System.
Compare the resume and job description.
Resume:
{parsed_resume}
Job Description:
{parsed_jd}
Provide:
1. Match percentage (0-100)
2. Matching skills
3. Missing skills
4. Strengths
5. Improvement suggestions
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# API ROUTE (PDF UPLOAD)
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Check if coming from PDF upload or text extraction
        if "resume" in request.files:
            # Original PDF upload flow
            resume_file = request.files["resume"]
            jd_text = request.form.get("job_description", "")
            
            if not jd_text:
                return jsonify({"error": "Job description is required"}), 400

            # Save and extract PDF
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
            resume_file.save(pdf_path)
            resume_text = extract_text_from_pdf(pdf_path)
        
        elif "resume_text" in request.files:
            # New text extraction flow from frontend
            resume_file = request.files["resume_text"]
            jd_text = request.form.get("job_description", "")
            
            resume_text = resume_file.read().decode('utf-8')
        else:
            return jsonify({"error": "Resume is required"}), 400

        if not resume_text or len(resume_text.strip()) == 0:
            return jsonify({"error": "Resume text is empty"}), 400

        # Parse using Gemini
        parsed_resume = parse_resume(resume_text)
        parsed_jd = parse_job_description(jd_text) if jd_text else ""

        # ATS Matching
        ats_result = ats_match(parsed_resume, parsed_jd) if jd_text else "No specific job description provided for matching."

        return jsonify({
            "success": True,
            "parsed_resume": parsed_resume,
            "parsed_job_description": parsed_jd,
            "ats_result": ats_result
        })
    
    except Exception as e:
        print(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8080)