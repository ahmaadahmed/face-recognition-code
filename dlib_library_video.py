#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib


# In[3]:


import os
import glob
import time


# In[4]:


import face_recognition


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[4]:


img = mpimg.imread(r"H:\face recognition\photos\amr zaki.jpg")
imgplot = plt.imshow(img)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


def face_rec(directory_path, img_path):
    # Specify the directory path where your photos are located
    directory_path = r"F:\face recognition\photos/"

    # Use glob to retrieve a list of file paths for all photos in the directory
    photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

    # Loop through the photo paths
    for photo_path in photo_paths:
        # Process each photo as needed
        # For example, you can load the photo using OpenCV and perform face recognition
        known_image = face_recognition.load_image_file(img_path)
        unknown_image = face_recognition.load_image_file(photo_path)

        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if(results==[True]):
            photo_path = os.path.split(photo_path)
            print(photo_path[1].split('.jpg')[0])

    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()
    return


# In[24]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test.jpg"
face_rec(directory_path, img)


# In[20]:


dir_path = r"F:\face recognition\test\\"


# In[21]:


import os


# In[26]:


directory_path = r"F:\face recognition\photos/"
# img = r"F:\face recognition\test\test1.jpg"
for i in os.listdir(dir_path):
    face_rec(directory_path, dir_path + i)


# In[18]:


def face_rec1(directory_path, img_path):
    # Use glob to retrieve a list of file paths for all photos in the directory
    photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

    # Process the known image
    known_image = face_recognition.load_image_file(img_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize variables to keep track of the nearest person and their distance
    nearest_person = None
    min_distance = float('inf')

    # Loop through the photo paths
    for photo_path in photo_paths:
        # Process each photo as needed
        unknown_image = face_recognition.load_image_file(photo_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        # Calculate the distance between face encodings
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        # Check if this person is closer than the current nearest person
        if distance < min_distance:
            nearest_person = os.path.splitext(os.path.basename(photo_path))[0]
            min_distance = distance

    # Print or display the nearest person information
    print(nearest_person)

    # Display the matched image
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.show()


# In[38]:


directory_path = r"F:\face recognition\photos/"
for i in os.listdir(dir_path):
    face_rec1(directory_path, dir_path + i)


# In[59]:


directory_path = r"F:\face recognition\photos/"
# img = r"F:\face recognition\test\test1.jpg"
for i in os.listdir(dir_path):
    face_rec1(directory_path, dir_path + i)


# In[61]:


start = time.time()
directory_path = r"F:\face recognition\photos/"
# img = r"F:\face recognition\test\test1.jpg"
for i in os.listdir(dir_path):
    face_rec1(directory_path, dir_path + i)
end = time.time()
print(end - start)


# In[64]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test26.jpg"
face_rec1(directory_path, img)


# In[65]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test27.jpg"
face_rec1(directory_path, img)


# In[66]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test28.jpg"
face_rec1(directory_path, img)


# In[67]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test29.jpg"
face_rec1(directory_path, img)


# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test30.jpg"
face_rec1(directory_path, img)


# In[20]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test30.jpg"
face_rec1(directory_path, img)


# In[ ]:





# In[22]:


dir_path = r"F:\face recognition\test\\"


# In[23]:


start = time.time()
directory_path = r"F:\face recognition\photos/"
for i in os.listdir(dir_path):
    face_rec1(directory_path, dir_path + i)
end = time.time()
print(end - start)


# In[ ]:


def face_rec1(directory_path, img_path):
    # Use glob to retrieve a list of file paths for all photos in the directory
    photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

    # Process the known image
    known_image = face_recognition.load_image_file(img_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize variables to keep track of the nearest person and their distance
    nearest_person = None
    min_distance = float('inf')

    # Loop through the photo paths
    for photo_path in photo_paths:
        # Process each photo as needed
        unknown_image = face_recognition.load_image_file(photo_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#         print(unknown_encoding)
        # Calculate the distance between face encodings
#         distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

#         # Check if this person is closer than the current nearest person
#         if distance < min_distance:
#             nearest_person = os.path.splitext(os.path.basename(photo_path))[0]
#             min_distance = distance

    # Print or display the nearest person information
#     print(nearest_person)
    print(known_encoding)
    print(100*"*")
    print(100*"*")
    print(100*"*")
    print(unknown_encoding)

    # Display the matched image
#     img = mpimg.imread(img_path)
#     imgplot = plt.imshow(img)
#     plt.show()


# In[ ]:





# In[49]:


def save_encoding_to_file(encoding, output_file_path):
    with open(output_file_path, 'w') as file:
        for value in encoding:
            file.write(str(value) + '\n')

def face_rec1(directory_path, save_directory):
    # Use glob to retrieve a list of file paths for all photos in the directory
    photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

    # Loop through the photo paths
    for photo_path in photo_paths:
        # Process each photo as needed
        unknown_image = face_recognition.load_image_file(photo_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        
        
        unknown_image_name = os.path.basename(file_path).split(".jpg")[0] 
        unknown_encoding_file_path = os.path.join(save_directory, f"{unknown_image_name}.txt")
        save_encoding_to_file(unknown_encoding, unknown_encoding_file_path)


# In[113]:


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


# In[114]:


# Specify the directory path for saving text files
save_directory = 'F:/face recognition/saved_text'

directory_path = r"F:\face recognition\photos/"
face_rec1(directory_path, save_directory)


# In[54]:


# # Specify the directory path for saving text files
# save_directory = 'F:/face recognition/encoding_test'

# directory_path = r"F:\face recognition\test/"
# face_rec1(directory_path, save_directory)


# In[91]:


import os
import face_recognition

def load_encodings_from_folder(folder_path):
    encodings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                encodings[file_name[:-4]] = [float(line.strip()) for line in lines]
    return encodings

def find_nearest_match(unknown_encoding, known_encodings):
    # Initialize variables for the nearest match
    best_match_name = None
    best_distance = float('inf')

    # Compare the unknown encoding with known encodings
    for name, known_encoding in known_encodings.items():
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        # Update the nearest match if a closer match is found
        if distance < best_distance:
            best_distance = distance
            best_match_name = name

    return best_match_name, best_distance

def recognize_face(unknown_encoding, known_encodings_folder):
    known_encodings = load_encodings_from_folder(known_encodings_folder)
    best_match_name, best_distance = find_nearest_match(unknown_encoding, known_encodings)

    return best_match_name, best_distance

# # Example usage
# unknown_image_path = r"F:\face recognition\test\test3.jpg"
# known_encodings_folder = r"F:\face recognition\saved_text"

# # Load the unknown encoding from the image
# unknown_image = face_recognition.load_image_file(unknown_image_path)
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# # Find the nearest match in the known encodings folder
# best_match_name, best_distance = recognize_face(unknown_encoding, known_encodings_folder)

# print(f"The nearest match is: {best_match_name} with a distance of {best_distance}")
# img = mpimg.imread(img_path)
# imgplot = plt.imshow(img)
# plt.show()


# In[118]:


# Example usage
# for i in os.listdir("F:\face recognition\test")
start = time.time()
folder = r"F:\face recognition\test"
for i in os.listdir(folder):
    start1 = time.time()
    unknown_image_path = folder +"\\"+ i
    known_encodings_folder = r"F:\face recognition\saved_text"

    # Load the unknown encoding from the image
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    # Find the nearest match in the known encodings folder
    best_match_name, best_distance = recognize_face(unknown_encoding, known_encodings_folder)

    print(f"The nearest match is: {best_match_name} with a distance of {best_distance}")
    img = mpimg.imread(unknown_image_path)
    imgplot = plt.imshow(img)
    plt.show()
    end1 = time.time()
    print(end1-start1)
    print(100*"*")
end = time.time()
print(end-start)


# In[95]:


# Example usage
# for i in os.listdir("F:\face recognition\test")
start = time.time()
folder = r"F:\face recognition\test2"
for i in os.listdir(folder):
    start1 = time.time()
    unknown_image_path = folder +"\\"+ i
    known_encodings_folder = r"F:\face recognition\saved_text"

    # Load the unknown encoding from the image
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    # Find the nearest match in the known encodings folder
    best_match_name, best_distance = recognize_face(unknown_encoding, known_encodings_folder)

    print(f"The nearest match is: {best_match_name} with a distance of {best_distance}")
    img = mpimg.imread(unknown_image_path)
    imgplot = plt.imshow(img)
    plt.show()
    end1 = time.time()
    print(end1-start1)
    print(100*"*")
end = time.time()
print(end-start)


# In[ ]:





# In[116]:


def load_encodings_from_folder(folder_path):
    encodings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                encodings[file_name[:-4]] = [float(line.strip()) for line in lines]
    return encodings

def find_nearest_match(unknown_encoding, known_encodings):
    best_match_name = None
    best_distance = float('inf')

    for name, known_encoding in known_encodings.items():
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        if distance < best_distance:
            best_distance = distance
            best_match_name = name

    return best_match_name, best_distance

def recognize_faces_in_video(video_path, known_encodings_folder):
    known_encodings = load_encodings_from_folder(known_encodings_folder)
    
    video_capture = cv2.VideoCapture(video_path)
    results = set()  # Use a set to store unique results

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB (as face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each face in the frame
        for face_encoding in face_encodings:
            # Find the nearest match for each face
            best_match_name, best_distance = find_nearest_match(face_encoding, known_encodings)
            results.add(best_match_name)

    video_capture.release()
    return results

# Example usage
video_path = r"C:\Users\pc\Documents\Bandicam\face1.mp4"
known_encodings_folder = r"F:\face recognition\saved_text"

results = recognize_faces_in_video(video_path, known_encodings_folder)

# # Print the results
# for result in results:
#     name, distance = result
#     print(f"Name: {name}, Distance: {distance}")


# In[86]:


results


# In[101]:


def load_encodings_from_folder(folder_path):
    encodings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                encodings[file_name[:-4]] = [float(line.strip()) for line in lines]
    return encodings

def find_nearest_match(unknown_encoding, known_encodings):
    best_match_name = None
    best_distance = float('inf')

    for name, known_encoding in known_encodings.items():
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        if distance < best_distance:
            best_distance = distance
            best_match_name = name

    return best_match_name, best_distance

def recognize_faces_in_video(video_path, known_encodings_folder):
    known_encodings = load_encodings_from_folder(known_encodings_folder)
    
    video_capture = cv2.VideoCapture(video_path)

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB (as face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each face in the frame
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Find the nearest match for each face
            best_match_name, best_distance = find_nearest_match(face_encoding, known_encodings)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Put the name on the rectangle
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, best_match_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the modified frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r"C:\Users\pc\Videos\Download Videos\emam.mp4"
known_encodings_folder = r"F:\face recognition\saved_text"

recognize_faces_in_video(video_path, known_encodings_folder)


# In[117]:


def load_encodings_from_folder(folder_path):
    encodings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                encodings[file_name[:-4]] = [float(line.strip()) for line in lines]
    return encodings

def find_nearest_match(unknown_encoding, known_encodings):
    best_match_name = None
    best_distance = float('inf')

    for name, known_encoding in known_encodings.items():
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        if distance < best_distance:
            best_distance = distance
            best_match_name = name

    return best_match_name, best_distance

def recognize_faces_in_video(video_path, known_encodings_folder, output_video_path):
    known_encodings = load_encodings_from_folder(known_encodings_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_capture.get(3)), int(video_capture.get(4)))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB (as face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each face in the frame
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Find the nearest match for each face
            best_match_name, best_distance = find_nearest_match(face_encoding, known_encodings)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Put the name on the rectangle
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, best_match_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Write the frame to the output video
        out.write(frame)

        # Display the modified frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r"C:\Users\pc\Documents\Bandicam\ekramy.mp4"
known_encodings_folder = r"F:\face recognition\saved_text"
output_video_path = r"C:\Users\pc\Documents\Bandicam\face3_output.mp4"

recognize_faces_in_video(video_path, known_encodings_folder, output_video_path)


# In[ ]:





# In[62]:


directory_path = r"F:\face recognition\photos/"
img = r"F:\face recognition\test\test24.jpg"
face_rec1(directory_path, img)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


start = time.time()

# Specify the directory path where your photos are located
directory_path = r"F:\face recognition\photos/"

# Use glob to retrieve a list of file paths for all photos in the directory
photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

# Loop through the photo paths
for photo_path in photo_paths:
    # Process each photo as needed
    # For example, you can load the photo using OpenCV and perform face recognition
    known_image = face_recognition.load_image_file(r"F:\face recognition\test\test15.jpg")
    unknown_image = face_recognition.load_image_file(photo_path)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    if(results==[True]):
        photo_path = os.path.split(photo_path)
        print(photo_path[1].split('.jpg')[0])

img = mpimg.imread(r"F:\face recognition\test\test15.jpg")
imgplot = plt.imshow(img)
plt.show()
# end = time.time()
# print(end - start)


# In[ ]:





# In[ ]:





# In[5]:


start = time.time()

# Specify the directory path where your photos are located
directory_path = r"H:\face recognition\photos/"

# Use glob to retrieve a list of file paths for all photos in the directory
photo_paths = glob.glob(directory_path + "*.jpg")  # Modify the file extension if needed

# Loop through the photo paths
for photo_path in photo_paths:
    # Process each photo as needed
    # For example, you can load the photo using OpenCV and perform face recognition
    known_image = face_recognition.load_image_file(r"H:\face recognition\test\test23.jpg")
    unknown_image = face_recognition.load_image_file(photo_path)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    if(results==[True]):
        photo_path = os.path.split(photo_path)
        print(photo_path[1].split('.jpg')[0])

img = mpimg.imread(r"H:\face recognition\test\test23.jpg")
imgplot = plt.imshow(img)
plt.show()
end = time.time()
print(end - start)

