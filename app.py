from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile
import traceback

# Import your existing ATS functions
from ats_functions import analyze_resume_jd_match, extract_text_from_resume

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files were submitted
        if 'resume' not in request.files or 'job_description' not in request.form:
            flash('No file or job description provided')
            return redirect(request.url)
        
        resume_file = request.files['resume']
        job_description = request.form['job_description']
        
        if resume_file.filename == '' or not job_description.strip():
            flash('No selected file or empty job description')
            return redirect(request.url)
        
        if resume_file and allowed_file(resume_file.filename):
            try:
                # Save the uploaded resume
                filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume_file.save(resume_path)
                
                # Analyze the resume and JD
                results = analyze_resume_jd_match(resume_path, job_description)
                
                if results:
                    return render_template('results.html', results=results)
                else:
                    flash('Analysis failed. Please check your files.')
                    return redirect(request.url)
            
            except Exception as e:
                flash(f'Error processing files: {str(e)}')
                print(traceback.format_exc())
                return redirect(request.url)
        
        else:
            flash('Allowed file types are PDF and DOCX')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.json
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing resume_text or job_description'}), 400
        
        results = analyze_resume_jd_match(
            resume_text=data['resume_text'],
            job_description=data['job_description']
        )
        
        if results:
            return jsonify(results)
        else:
            return jsonify({'error': 'Analysis failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)