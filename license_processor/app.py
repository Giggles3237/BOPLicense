import streamlit as st
import cv2
import numpy as np
import os
from process_license import (
    four_point_transform, 
    resize_to_license_dimensions,
    create_pdf
)

def process_image(image, threshold_value):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binary threshold
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Draw contours for debugging
    debug_contours = image.copy()
    cv2.drawContours(debug_contours, contours[:5], -1, (0,255,0), 2)
    
    # Process largest contours
    detection = image.copy()
    box = None
    
    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]
        area_ratio = area / img_area
        
        if 0.05 <= area_ratio <= 0.95:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            width = rect[1][0]
            height = rect[1][1]
            if width < height:
                width, height = height, width
            
            aspect_ratio = width / height
            
            if 1.2 <= aspect_ratio <= 2.0:
                cv2.drawContours(detection, [box], 0, (0,255,0), 2)
                break
    
    return binary, debug_contours, detection, box

def round_corners(image, radius=30):
    # Create a copy of the image
    result = image.copy()
    
    # Get dimensions
    h, w = image.shape[:2]
    
    # Create a mask with rounded corners
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw filled white rectangle with rounded corners
    cv2.rectangle(mask, (radius, 0), (w-radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h-radius), 255, -1)
    cv2.ellipse(mask, (radius, radius), (radius, radius), 180, 0, 90, 255, -1)
    cv2.ellipse(mask, (w-radius, radius), (radius, radius), 270, 0, 90, 255, -1)
    cv2.ellipse(mask, (radius, h-radius), (radius, radius), 90, 0, 90, 255, -1)
    cv2.ellipse(mask, (w-radius, h-radius), (radius, radius), 0, 0, 90, 255, -1)
    
    # Create white background
    white_bg = np.ones_like(image) * 255
    
    # Convert mask to 3 channel
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Blend the image with the background using the mask
    result = np.where(mask_3channel == 255, image, white_bg)
    
    return result

def rotate_image(image, key):
    # Initialize rotation state for this specific key if it doesn't exist
    if f'rotation_{key}' not in st.session_state:
        st.session_state[f'rotation_{key}'] = 0
    
    # Update rotation angle for this specific image only
    st.session_state[f'rotation_{key}'] = (st.session_state[f'rotation_{key}'] + 90) % 360
    angle = st.session_state[f'rotation_{key}']
    
    # Create a copy of the image to avoid modifying the original
    result = image.copy()
    
    # Apply rotation based on this specific image's angle
    if angle == 90:
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        result = cv2.rotate(result, cv2.ROTATE_180)
    elif angle == 270:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    
    return result

def create_downloadable_pdf(front_path, back_path=None, filename="license.pdf"):
    # Create the PDF
    if back_path is None:
        # Single image mode - only use front image
        create_pdf(front_path, front_path, filename, single_page=True)
    else:
        # Dual image mode - use both front and back
        create_pdf(front_path, back_path, filename, single_page=False)
    
    # Read the PDF file
    with open(filename, "rb") as f:
        pdf_bytes = f.read()
    
    # Remove the temporary PDF file
    os.remove(filename)
    
    return pdf_bytes

def main():
    st.title("Card Adjuster")
    
    tab1, tab2 = st.tabs(["Single Image", "Front & Back"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                threshold = st.number_input("Threshold", min_value=0, max_value=255, value=120, step=1, key="thresh_single")
            with col2:
                if st.button("Rotate Image", key="rotate_single"):
                    image = rotate_image(image, "single")
            
            binary, contours, detection, box = process_image(image, threshold)
            
            if box is not None:
                warped = four_point_transform(image, box.astype("float32"))
                final = resize_to_license_dimensions(warped)
                final_rounded = round_corners(final)
                st.image(cv2.cvtColor(final_rounded, cv2.COLOR_BGR2RGB))
                
                if st.button("Export to PDF", key="pdf_single"):
                    try:
                        cv2.imwrite("temp_single.png", final_rounded)
                        pdf_bytes = create_downloadable_pdf("temp_single.png", filename="license_single.pdf")
                        
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name="license_single.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
                    finally:
                        if os.path.exists("temp_single.png"):
                            os.remove("temp_single.png")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Front Image")
            front_file = st.file_uploader("Choose front image...", type=["jpg", "jpeg", "png"], key="front")
            if front_file is not None:
                front_bytes = np.asarray(bytearray(front_file.read()), dtype=np.uint8)
                front_image = cv2.imdecode(front_bytes, cv2.IMREAD_COLOR)
                
                fcol1, fcol2 = st.columns([3, 1])
                with fcol1:
                    front_threshold = st.number_input("Threshold", min_value=0, max_value=255, value=120, step=1, key="thresh_front")
                with fcol2:
                    if st.button("Rotate Image", key="rotate_front"):
                        front_image = rotate_image(front_image, "front")
                
                front_binary, front_contours, front_detection, front_box = process_image(
                    front_image, front_threshold
                )
                
                if front_box is not None:
                    front_warped = four_point_transform(front_image, front_box.astype("float32"))
                    front_final = resize_to_license_dimensions(front_warped)
                    front_rounded = round_corners(front_final)
                    st.image(cv2.cvtColor(front_rounded, cv2.COLOR_BGR2RGB))
        
        with col2:
            st.markdown("### Back Image")
            back_file = st.file_uploader("Choose back image...", type=["jpg", "jpeg", "png"], key="back")
            if back_file is not None:
                back_bytes = np.asarray(bytearray(back_file.read()), dtype=np.uint8)
                back_image = cv2.imdecode(back_bytes, cv2.IMREAD_COLOR)
                
                bcol1, bcol2 = st.columns([3, 1])
                with bcol1:
                    back_threshold = st.number_input("Threshold", min_value=0, max_value=255, value=120, step=1, key="thresh_back")
                with bcol2:
                    if st.button("Rotate Image", key="rotate_back"):
                        back_image = rotate_image(back_image, "back")
                
                back_binary, back_contours, back_detection, back_box = process_image(
                    back_image, back_threshold
                )
                
                if back_box is not None:
                    back_warped = four_point_transform(back_image, back_box.astype("float32"))
                    back_final = resize_to_license_dimensions(back_warped)
                    back_rounded = round_corners(back_final)
                    st.image(cv2.cvtColor(back_rounded, cv2.COLOR_BGR2RGB))
        
        # PDF export button (only show if both images are processed successfully)
        if front_file is not None and back_file is not None and front_box is not None and back_box is not None:
            if st.button("Export to PDF", key="pdf_dual"):
                try:
                    cv2.imwrite("temp_front.png", front_rounded)
                    cv2.imwrite("temp_back.png", back_rounded)
                    pdf_bytes = create_downloadable_pdf("temp_front.png", "temp_back.png", "license_dual.pdf")
                    
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name="license_dual.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error creating PDF: {str(e)}")
                finally:
                    for temp_file in ["temp_front.png", "temp_back.png"]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

if __name__ == "__main__":
    main() 