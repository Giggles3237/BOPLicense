import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def order_points(pts):
    # Order the points in the following order:
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    # Compute width and height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Destination points: a straight-on rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def enhance_contrast(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def find_license_card(image, prefix=""):
    # Save original for debugging
    debug_original = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Generate contour images for thresholds from 160 to 300
    for thresh_val in range(160, 301, 20):
        # Find contours for this threshold
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Draw contours for debugging
        debug_contours = debug_original.copy()
        cv2.drawContours(debug_contours, contours[:5], -1, (0,255,0), 2)
        cv2.imwrite(f"{prefix}debug_4_contours_{thresh_val}.png", debug_contours)
        
        # Process contours for license detection
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
                    return order_points(box.astype("float32"))
    
    raise ValueError("No license found in the image.")

def resize_to_license_dimensions(image, dpi=300):
    # Standard license size: 3.375" x 2.125"
    width_inch = 3.375
    height_inch = 2.125
    width_px = int(width_inch * dpi)
    height_px = int(height_inch * dpi)
    resized = cv2.resize(image, (width_px, height_px), interpolation=cv2.INTER_LINEAR)
    return resized

def create_pdf(front_image_path, back_image_path, output_filename):
    c = canvas.Canvas(output_filename, pagesize=letter)

    # Actual license dimensions in PDF units
    license_width = 3.375 * inch
    license_height = 2.125 * inch

    # Place front image
    c.drawImage(front_image_path, x=1*inch, y=7*inch, width=license_width, height=license_height)
    # Place back image below front for demonstration
    c.drawImage(back_image_path, x=1*inch, y=(7 - 2.5)*inch, width=license_width, height=license_height)

    c.showPage()
    c.save()

if __name__ == "__main__":
    # Process front image
    front_input_image = cv2.imread("front_photo.jpg")
    front_rect = find_license_card(front_input_image)
    front_warped = four_point_transform(front_input_image, front_rect)
    # Check orientation and rotate if needed
    if front_warped.shape[0] > front_warped.shape[1]:
        front_warped = cv2.rotate(front_warped, cv2.ROTATE_90_CLOCKWISE)
    front_final = resize_to_license_dimensions(front_warped)
    cv2.imwrite("front_final.png", front_final)

    # Process back image
    back_input_image = cv2.imread("back_photo.jpg")
    back_rect = find_license_card(back_input_image)
    back_warped = four_point_transform(back_input_image, back_rect)
    # Check orientation and rotate if needed
    if back_warped.shape[0] > back_warped.shape[1]:
        back_warped = cv2.rotate(back_warped, cv2.ROTATE_90_CLOCKWISE)
    back_final = resize_to_license_dimensions(back_warped)
    cv2.imwrite("back_final.png", back_final)

    # Create the PDF
    create_pdf("front_final.png", "back_final.png", "license_output.pdf")
    print("PDF created successfully!")
