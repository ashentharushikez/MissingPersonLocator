import pickle

with open("face_encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)
    print("Names:", known_face_names)
    print("Number of Encodings:", len(known_face_encodings))
