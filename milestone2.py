import os
import cv2
import pandas as pd

# Paths to Milestone 2 and Template images
milestone2_folder = "D:/KlA_Unhack/DataSet/Milestone 2"
template_folder = "D:/KlA_Unhack/DataSet/Template images"
output_folder = "D:/KlA_Unhack/DataSet/Output-Milestone2"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of image files in Milestone 2 folder
milestone2_images = [f for f in os.listdir(milestone2_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

# Function to rotate an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Prepare the output data
output_data = []

# Perform template matching for each image in Milestone 2
for image_file in milestone2_images:
    image_path = os.path.join(milestone2_folder, image_file)
    image_gray = load_image_grayscale(image_path)
    
    detected_speed = None
    max_val = 0.5  # A threshold value to determine if the template match is significant
    
    for speed, template in templates.items():
        # Check the original image
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        _, curr_max_val, _, _ = cv2.minMaxLoc(res)
        if curr_max_val > max_val:
            max_val = curr_max_val
            detected_speed = speed
        
        # Perform matching on rotated and blurred images
        for angle in range(-15, 16, 5):  # Rotate from -15 to 15 degrees
            rotated_image = rotate_image(image_gray, angle)
            
            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(rotated_image, (5, 5), 0)
            
            # Match template on rotated and blurred image
            res = cv2.matchTemplate(blurred_image, template, cv2.TM_CCOEFF_NORMED)
            _, curr_max_val, _, _ = cv2.minMaxLoc(res)
            
            if curr_max_val > max_val:
                max_val = curr_max_val
                detected_speed = speed

        # Also match on the original blurred image
        blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)
        res = cv2.matchTemplate(blurred_image, template, cv2.TM_CCOEFF_NORMED)
        _, curr_max_val, _, _ = cv2.minMaxLoc(res)
        
        if curr_max_val > max_val:
            max_val = curr_max_val
            detected_speed = speed
    
    output_data.append([image_file, detected_speed])

# Create a DataFrame and save to CSV
df_output = pd.DataFrame(output_data, columns=["Input image name", "Speed limit"])
output_csv_path = os.path.join(output_folder, 'M2_Output.csv')
df_output.to_csv(output_csv_path, index=False)

print(f"Output CSV file saved at: {output_csv_path}")
