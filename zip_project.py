import os
import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        # Exclude large and unnecessary directories
        dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__', 'scratch', '.vscode', 'node_modules')]
        for file in files:
            if file == 'atomizer-plus-ultra.zip' or file == 'zip_project.py':
                continue
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, path)
            ziph.write(file_path, arcname)

if __name__ == '__main__':
    desktop_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join(desktop_path, 'hades_webapp_final.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('.', zipf)
    print(f"Project zipped to {zip_path}")
