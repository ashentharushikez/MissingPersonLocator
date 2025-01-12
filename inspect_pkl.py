import pickle

with open("face_recognition_model.pkl", "rb") as f:
    data = pickle.load(f)

# Check the overall structure
print("Data type:", type(data))

# If it's a dictionary, check its keys
if isinstance(data, dict):
    print("Keys:", data.keys())

    # Display a sample encoding and its corresponding name
    encodings = data.get("encodings", [])
    labels = data.get("labels", [])
    
    print("Number of encodings:", len(encodings))
    print("Number of labels:", len(labels))
    if encodings and labels:
        print("Sample encoding (first 5 values):", encodings[0][:5])  # Show first 5 values of the first encoding
        print("Sample label:", labels[0])
else:
    print("Contents:", data)  # If it's not a dictionary, print the content directly
