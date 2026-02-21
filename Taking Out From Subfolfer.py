import os
import shutil

# Define the path to the main sessions directory
sessions_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\The_Child_Affective_Facial_Expression_(CAFE)_set\sessions"

# Iterate over items in the sessions_path
for folder_name in os.listdir(sessions_path):
    folder_path = os.path.join(sessions_path, folder_name)

    # Check if the item is a subfolder
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            source_file = os.path.join(folder_path, file_name)
            destination_file = os.path.join(sessions_path, file_name)

            # If a file with the same name exists, rename to avoid overwrite
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(sessions_path, f"{base}_{counter}{ext}")
                    counter += 1

            # Move the file
            shutil.move(source_file, destination_file)

        # Remove the now-empty folder
        os.rmdir(folder_path)

print("All files have been moved and subfolders deleted.")
