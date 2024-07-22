import os
import cv2
import pandas as pd

# Paths to Milestone 1 and Template images
milestone1_folder = "D:/KlA_Unhack/DataSet/Milestone 1"
template_folder = "D:/KlA_Unhack/DataSet/Template images"
output_folder = "D:/KlA_Unhack/DataSet/Output-Milestone1"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of image files in Milestone 1 folder
milestone1_images = [f for f in os.listdir(milestone1_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Get the list of template images
template_images = [f for f in os.listdir(template_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function to load and convert image to grayscale
def load_image_grayscale(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

# Load templates
templates = {}
for template_file in template_images:
    template_name = os.path.splitext(template_file)[0].split('-')[1]
    template_path = os.path.join(template_folder, template_file)
    templates[template_name] = load_image_grayscale(template_path)

# Prepare the output data
output_data = []

# Perform template matching for each image in Milestone 1
for image_file in milestone1_images:
    image_path = os.path.join(milestone1_folder, image_file)
    image_gray = load_image_grayscale(image_path)
    
    detected_speed = None
    max_val = 0.5  # A threshold value to determine if the template match is significant
    
    for speed, template in templates.items():
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, curr_max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if curr_max_val > max_val:
            max_val = curr_max_val
            detected_speed = speed
    
    output_data.append([image_file, detected_speed])

# Create a DataFrame and save to CSV
df_output = pd.DataFrame(output_data, columns=["Input image name", "Speed limit"])
output_csv_path = os.path.join(output_folder, 'M1_Output.csv')
df_output.to_csv(output_csv_path, index=False)

print(f"Output CSV file saved at: {output_csv_path}")
