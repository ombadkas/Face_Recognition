import cv2
import os

def apply_augmentation(image_path, output_folder):
    original_image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    image_name_no_ext, image_ext = os.path.splitext(image_name)

    hue_shifts = [0, 10, 20, 30]
    rotation_angles = [-20,-10,-5,0,5,10,15,20,25,30, 60, 90]

    # Apply counterclockwise rotation and hue shift
    for angle in rotation_angles:
        for hue_shift in hue_shifts:
            # Counterclockwise Rotation
            rotation_matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), -angle, 1)
            rotated_image = cv2.warpAffine(original_image, rotation_matrix, (original_image.shape[1], original_image.shape[0]))

            # Hue shift
            hsv_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
            augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            # Convert to grayscale
            gray_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

            # Save augmented image
            output_file = os.path.join(output_folder, f'{image_name_no_ext}_rotated_{angle}_hue_{hue_shift}_gray{image_ext}')
            cv2.imwrite(output_file, gray_image)



input_folder = "F:/pythonProject/face_detection/valid/unknown"

output_folder = "F:/pythonProject/face_detection/valid/unknown"

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        apply_augmentation(image_path, output_folder)
