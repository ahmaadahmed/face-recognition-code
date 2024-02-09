import os
import glob
import face_recognition


def save_encoding_to_file(encoding, output_file_path):
    with open(output_file_path, 'w') as file:
        for value in encoding:
            file.write(str(value) + '\n')

def face_rec1(directory_path, save_directory):
    # Use glob to retrieve a list of file paths for all photos in the directory
    photo_paths = glob.glob(os.path.join(directory_path, "*.jpg"))  # Modify the file extension if needed

    # Loop through the photo paths
    for photo_path in photo_paths:
        # Process each photo as needed
        unknown_image = face_recognition.load_image_file(photo_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) > 0:
            # Take the first face encoding if multiple faces are detected
            unknown_encoding = unknown_encodings[0]

            # Get the name of the image file (without extension)
            unknown_image_name = os.path.basename(photo_path).split(".jpg")[0]

            # Construct the path for the output text file
            unknown_encoding_file_path = os.path.join(save_directory, f"{unknown_image_name}.txt")

            # Save the face encoding to the text file
            save_encoding_to_file(unknown_encoding, unknown_encoding_file_path)


# Specify the directory path for saving text files
save_directory = r'C:\Users\work\Downloads\face-recognition-code-master\saved_text1'

directory_path = r"C:\Users\work\Downloads\face-recognition-code-master\images/"
face_rec1(directory_path, save_directory)