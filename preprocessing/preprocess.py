import os
import shutil

# Specifying the source directory of the FER-2013 dataset on the local machine
SOURCE_DIR = r'C:\Users\aayus\Desktop\UJ\Year Project\HYP\archive'
# Specifying the destination directory in the project file structure for the dataset
DEST_DIR = 'data/processed'

# Categorising the dataset emotions into attentive or inattentive
ATTENTIVE = {'happy', 'neutral', 'surprise'}
INATTENTIVE = {'angry', 'disgust', 'fear', 'sad'}

def classify_emotion(emotion):
    if emotion in ATTENTIVE:
        return 'attentive'
    elif emotion in INATTENTIVE:
        return 'inattentive'
    return None

def process_dataset(split):
    input_path = os.path.join(SOURCE_DIR, split)
    output_path = os.path.join(DEST_DIR, split)
    
    for emotion_folder in os.listdir(input_path):
        class_label = classify_emotion(emotion_folder)
        if class_label is None:
            continue
        
        src_folder = os.path.join(input_path, emotion_folder)
        dest_folder = os.path.join(output_path, class_label)
        os.makedirs(dest_folder, exist_ok=True)
        
        for file in os.listdir(src_folder):
            shutil.copy2(os.path.join(src_folder, file), os.path.join(dest_folder, file))
            
        print(f"Copied {emotion_folder} to {class_label}")
        
def main():
    for split in ['train', 'test']:
        process_dataset(split)
        
if __name__ == "__main__":
    main()