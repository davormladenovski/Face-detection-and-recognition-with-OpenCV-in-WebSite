import mimetypes
import os
from email.mime.text import MIMEText
import smtplib
from flask import Flask, flash, render_template, render_template_string, request, redirect, url_for
import cv2
from werkzeug.utils import secure_filename
import math 
import numpy as np
import face_recognition
from PIL import Image
from moviepy.editor import VideoFileClip


app = Flask(__name__, static_url_path='/static')

app.config["UPLOAD_FOLDER"] = os.path.join('static', 'Videos')
app.config["IMAGE_UPLOADS"] = os.path.join('static', 'Images')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    known_faces_encodings, known_face_names = encode_known_faces()

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = None
        confidence = None

        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])


        if name is not None and confidence is not None:
            cv2.rectangle(img, (left, top), (right, bottom), (157, 102, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (157, 102, 0), cv2.FILLED)
            cv2.putText(img, f"{name} ({confidence})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (left, top), (right, bottom), (157, 102, 0), 2)
            cv2.rectangle(img, (left, bottom), (right, bottom), (157, 102, 0))

            

    return img, len(face_locations)

def detect_faces_video(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        return None
    
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    known_faces_encodings, known_face_names = encode_known_faces()

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            name = None
            confidence = None

            face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            if name is not None and confidence is not None:
                cv2.rectangle(frame, (left, top), (right, bottom), (157, 102, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (157, 102, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence})", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (157, 102, 0), 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom), (157, 102, 0))

        out.write(frame)

    out.release()
    video_capture.release()
    cv2.destroyAllWindows()

    print("Processing complete. Video saved to:", output_path)

def encode_known_faces():
    known_faces_encodings = []
    known_face_names = []
    
    for image in os.listdir('static/faces'):
        face_image = face_recognition.load_image_file(f'static/faces/{image}')
        face_encoding = face_recognition.face_encodings(face_image)[0]
        
        known_faces_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(image)[0])
    
    return known_faces_encodings, known_face_names

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_value = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_value * 100, 2)) + '%'
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


@app.route("/home", methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST': 
        if 'file' not in request.files: #ako baranjeto go nema fajlot se vrakajme na istiot url
            return redirect(request.url)

        file = request.files['file'] #se zema fajlot od baranjeto

        if file.filename == '' or not allowed_file(file.filename): #proverka za validnosta na fajlot i negoviot format
            return render_template("Home.html", error_message="Invalid file format. Please upload a photo or video.")

        filename = secure_filename(file.filename) #go pravime fajlot siguren za ponatamosnata obrabotka

        if filename.endswith(('mp4', 'avi', 'mov')):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            base_filename = os.path.splitext(filename)[0]
            output_video_path = os.path.join(app.config["UPLOAD_FOLDER"], 'processed_' + base_filename + '.avi')
            
            detect_faces_video(filepath, output_video_path)

            video_clip = VideoFileClip(output_video_path)
            output_mp4_path = os.path.join(app.config["UPLOAD_FOLDER"], 'processed_' + base_filename + '.mp4')
            video_clip.write_videofile(output_mp4_path, codec='libx264')

            video_clip = VideoFileClip(output_mp4_path)
            width = int(video_clip.size[0])
            height = int(video_clip.size[1])
            duration = int(video_clip.duration)
            
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            file_size = os.path.getsize(filepath)
            
            file_type = os.path.splitext(filepath)[1][1:].upper()

            return render_template("video_detection.html", video_filename='processed_' + base_filename + '.mp4',width=width,
                                   height=height,duration=duration,fps=fps,
                                   frame_count=frame_count,file_size=file_size,
                                   file_type=file_type)
        
        elif filename.endswith(('png', 'jpg', 'jpeg', 'gif')):

            filepath = os.path.join(app.config["IMAGE_UPLOADS"], filename)
            file.save(filepath)

            modified_img, num_faces = detect_faces_image(filepath)

            if modified_img is None:
                return render_template("Home.html", error_message="Error loading or processing the image")

            cv2.imwrite(filepath, modified_img)

            image = Image.open(filepath)
            width, height = image.size
            color_space = image.mode

            file_size = os.path.getsize(filepath)

            file_type = mimetypes.guess_type(filepath)[0]

            return render_template("detection.html", filename=filename, 
                                   width=width, height=height,color_space=color_space, 
                                   file_size=file_size, file_type=file_type, num_faces=num_faces)
        else:
            return render_template("Home.html", error_message="Unsupported file format.")

    return render_template("Home.html")

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'dmladenovski79@gmail.com'
EMAIL_PASSWORD = 'lnbj zosw byim owzx'

@app.route('/send_email', methods=['POST'])
def send_email():
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']
    
    msg = MIMEText(f"From: {email},\nMessage: \n"+message)
    msg['Subject'] = subject
    msg['From'] = email
    msg['To'] = EMAIL_ADDRESS
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls() #otvaranje bezbedna komunikacija
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg) #prakanje na porakata
        flash("Your message sent successfully, expect a quick response from us! ", 'success')
    except Exception as e:
        flash(f"Failed to send email: {str(e)}", 'danger')

    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template_string(open('Ho').read())

if __name__ == '__main__':
    app.run(debug=True)
    
