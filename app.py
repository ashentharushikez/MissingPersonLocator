from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import pickle

# Start a Flask app
app = Flask(__name__)
CORS(app)  # Allow access from other domains

# Load face data and model from a file
print("Loading face data and model...")
with open("face_recognition_model.pkl", 'rb') as f:
    data = pickle.load(f)  # Load saved data
    known_face_encodings = data["encodings"]  # Pre-saved face data
    known_face_names = data["labels"]  # Names for faces
    model = data["model"]  # Machine learning model
print("Face data and model loaded.")

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Show the main webpage

# Route to handle uploaded images
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:  # Check if a file is uploaded
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']  # Get the uploaded file
    uploaded_image = face_recognition.load_image_file(file)  # Load the image

    # Find faces in the image
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)
    
    face_matches = []  # Store results

    # Match faces in the image with known faces
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)  # Compare with known faces
        best_match_index = np.argmin(distances)  # Find the closest match
        name = "Unknown"  # Default name if no match

        # If a close match is found, use the name
        if distances[best_match_index] < 0.6:  # Adjust this number to control matching
            name = known_face_names[best_match_index].replace("pins_", "")  # Clean up name format
        
        face_matches.append(name)  # Add name to results

    return jsonify({"faces": face_matches})  # Return the results as JSON

# Test route to check if the app works
@app.route('/test')
def test():
    return "Flask is working!"  # Simple check for the app

# Start the app
if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Run on port 5001
