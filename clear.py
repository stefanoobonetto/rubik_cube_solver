import os

def clear_folder(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

folder_paths = ['scan', 'squares/1', 'squares/2', 'squares/3','squares/4', 'squares/5', 'squares/6', 'faces']

for path in folder_paths:
    clear_folder(path)

