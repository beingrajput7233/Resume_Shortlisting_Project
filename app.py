import os
import shutil
import shortuuid
from werkzeug.utils import secure_filename
from flask import Flask,  jsonify, request,  render_template
from flask_cors import CORS
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf','docx','doc'}

import shortlist

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
cf_port = os.getenv("PORT")
 
# allowed files
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
 # make tree function for converting to the json format
def make_tree(path):
     tree = dict(name=path, children=[])
     try: lst = os.listdir(path)
     except OSError:
         pass #ignore errors
     else:
         for name in lst:
             fn = os.path.join(path, name)
             if os.path.isdir(fn):
                 tree['children'].append(make_tree(fn))
             else:
                 tree['children'].append(dict(name=fn))
     return tree
 
 
 
 
 # basic router for the file
@app.route('/')
def file_upload():
    return render_template('upload.html')
 
 
 # check the updated file on the server
@app.route('/list')
def list_files():
    return make_tree(r'uploads')
 	
# file uploader function
@app.route('/uploader', methods = ['POST'])
def upload_file():
     try:
     # check for file
     # print (request.files)
         if  'resume' not in request.files:
             # send error message
             return jsonify({'message': 'No file selected for jobdesc or resume'}), 400
 
         # resumes
         resumes = request.files.getlist('resume')
 
         # read the input query i.e. job description
         query = request.form['query']
         # generate a random string for the folder name        
         rand_str = shortuuid.uuid()
         folderp1 = os.path.join(app.config['UPLOAD_FOLDER'], rand_str,'raw_resume')
         folderp2 = os.path.join(app.config['UPLOAD_FOLDER'], rand_str,'txt_resume')
         os.makedirs(folderp1)
         os.makedirs(folderp2)
 
         # process all resumes
         for resume in resumes:
             # check if file is allowed
             # print(resume)
             if resume.filename == '':
                 return jsonify({'message': 'No file selected for resume'}), 400
             if resume.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                 return jsonify({'message': 'File extension not allowed'}), 400
 
             # save file
             resume.save(os.path.join(app.config['UPLOAD_FOLDER'] +'/'+ rand_str+ '/raw_resume', secure_filename(resume.filename)))
 
         # result json
         result = ""
 
         # call the shortlist function
         result,query_skills= shortlist.toTxt(query,rand_str)
         # shutil.rmtree(app.config['UPLOAD_FOLDER'] + '/' + rand_str)
         
         skills = []
         for i in query_skills:
             skills.append({'skill':i})
 
         return jsonify({'results': result,'querySkills':skills}), 200