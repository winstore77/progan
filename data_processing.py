# Loading the image file
def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    return pixels
    
# extract the face from a loaded image and resize
def extract_face(model, pixels, required_size=(128, 128)):
    # detect face in the image
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None
    
    # extract details of the face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    
    x2, y2 = x1 + width, y1 + height
    face_pixels = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(required_size)
    face_array = asarray(image)
    
    return face_array
    
# load images and extract faces for all images in a directory
def load_faces(directory, n_faces):
    # prepare model
    model = MTCNN()
    faces = list()
    
    for filename in os.listdir(directory):
        # Computing the retrieval and extraction of faces
        pixels = load_image(directory + filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
        print(len(faces), face.shape)
        if len(faces) >= n_faces:
            break
            
    return asarray(faces)
