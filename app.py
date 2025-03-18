import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf','docx','doc'}
 
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
 	
 	
if __name__ == '__main__':
 	    if cf_port is None:
 		    app.run(host='0.0.0.0', port=5000, debug=True)
       
        else:
 		    app.run(host='0.0.0.0', port=int(cf_port), debug=False)