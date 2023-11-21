import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def estimate_body_mass(image):
    # Use MediaPipe Pose to estimate body landmarks
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate and display body mass percentage
            height, width, _ = image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3  # Adjust the font scale for a smaller size
            font_color = (0, 255, 0)  # Green color
            font_thickness = 1  # Adjust the font thickness

            # Replace this line with your actual body mass percentage calculation
            body_mass_percentage = 10

            text = f"Body Mass Percentage: {body_mass_percentage}%"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = height - 10  # Adjust the vertical position

            # Highlighted background
            background_color = (0, 0, 255)  # Red color
            background_height = text_size[1] + 4  # Add a small margin
            cv2.rectangle(image, (text_x, text_y - background_height), (text_x + text_size[0], text_y), background_color, thickness=cv2.FILLED)

            # Draw text
            cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            return image
        else:
            st.warning("No pose landmarks detected.")
            return None

# Custom streamlit component for displaying result image
def st_image_with_opencv(result_image, caption="Result"):
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption=caption, use_column_width=True)

def main():
    st.title("Body Mass Estimation App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Estimate body landmarks
        result_image = estimate_body_mass(image)

        if result_image is not None:
            # Display the result image using the custom component
            st_image_with_opencv(result_image, caption="Result")

if __name__ == "__main__":
    main()
