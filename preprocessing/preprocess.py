import os
import shutil

# Absolute path to the original FER-2013 dataset
SOURCE_DIR = r'C:\Users\aayus\Desktop\UJ\Year Project\HYP\archive'

# Destination path within your project to store the processed dataset
DEST_DIR = 'data/processed'

# Categorising the dataset emotions into attentive or inattentive labels
ATTENTIVE = {'happy', 'neutral', 'surprise'}
INATTENTIVE = {'angry', 'disgust', 'fear', 'sad'}

# Function to classify an emotion name into attentive, inattentive or none
def classify_emotion(emotion):
    if emotion in ATTENTIVE:
        return 'attentive'
    elif emotion in INATTENTIVE:
        return 'inattentive'
    return None

# Processes either the training or test split by relabelling and copying files
def process_dataset(split):
    input_path = os.path.join(SOURCE_DIR, split)
    output_path = os.path.join(DEST_DIR, split)
    
    # Loop over each emotion folder in the split
    for emotion_folder in os.listdir(input_path):
        class_label = classify_emotion(emotion_folder)
        if class_label is None:
            continue
        
        # Construct full paths for source and destination directories
        src_folder = os.path.join(input_path, emotion_folder)
        dest_folder = os.path.join(output_path, class_label)
        
        # Create the destination folder if it does not exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy each image file from the original emotion folder to the new class folder
        for file in os.listdir(src_folder):
            shutil.copy2(os.path.join(src_folder, file), os.path.join(dest_folder, file))
        
        # Log which folder was copied and to which class
        print(f"Copied {emotion_folder} to {class_label}")

# Entry point of the script for processing both the train and test splits
def main():
    for split in ['train', 'test']:
        process_dataset(split)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()