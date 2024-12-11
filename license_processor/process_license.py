import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

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
    cv2.imwrite(f"{prefix}debug_1_gray.png", gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try a wider range of threshold values
    threshold_values = [80, 120, 160, 180, 200, 220, 240]
    
    for thresh_val in threshold_values:
        print(f"Trying threshold value: {thresh_val}")
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{prefix}debug_2_binary_{thresh_val}.png", binary)
        
        # Dilate to connect edges
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        cv2.imwrite(f"{prefix}debug_3_dilated_{thresh_val}.png", dilated)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Draw all contours for debugging
        debug_contours = debug_original.copy()
        cv2.drawContours(debug_contours, contours[:5], -1, (0,255,0), 2)
        cv2.imwrite(f"{prefix}debug_4_contours_{thresh_val}.png", debug_contours)
        
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            img_area = image.shape[0] * image.shape[1]
            area_ratio = area / img_area
            
            print(f"  Checking contour with area ratio: {area_ratio:.3f}")
            
            # More lenient area check
            if 0.05 <= area_ratio <= 0.95:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                width = rect[1][0]
                height = rect[1][1]
                if width < height:
                    width, height = height, width
                
                aspect_ratio = width / height
                print(f"  Aspect ratio: {aspect_ratio:.2f}")
                
                # More lenient aspect ratio check
                if 1.2 <= aspect_ratio <= 2.0:
                    debug_detection = debug_original.copy()
                    cv2.drawContours(debug_detection, [box], 0, (0,255,0), 2)
                    cv2.imwrite(f"{prefix}debug_5_detection_{thresh_val}.png", debug_detection)
                    
                    print(f"Found license with threshold {thresh_val}")
                    print(f"Final aspect ratio: {aspect_ratio:.2f}")
                    print(f"Final area ratio: {area_ratio:.3f}")
                    
                    return order_points(box.astype("float32"))
    
    # If no match found with normal thresholding, try inverse (THRESH_BINARY_INV)
    print("Trying inverse thresholding...")
    for thresh_val in threshold_values:
        print(f"Trying inverse threshold value: {thresh_val}")
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"{prefix}debug_2_binary_inv_{thresh_val}.png", binary)
        
        # Same processing as above...
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        cv2.imwrite(f"{prefix}debug_3_dilated_inv_{thresh_val}.png", dilated)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        debug_contours = debug_original.copy()
        cv2.drawContours(debug_contours, contours[:5], -1, (0,255,0), 2)
        cv2.imwrite(f"{prefix}debug_4_contours_inv_{thresh_val}.png", debug_contours)
        
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            img_area = image.shape[0] * image.shape[1]
            area_ratio = area / img_area
            
            print(f"  Checking inverse contour with area ratio: {area_ratio:.3f}")
            
            if 0.05 <= area_ratio <= 0.95:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                width = rect[1][0]
                height = rect[1][1]
                if width < height:
                    width, height = height, width
                
                # More lenient checks for license dimensions
                if (1.4 <= aspect_ratio <= 1.9 and      # Standard ratio is ~1.588
                    0.05 <= (area/img_area) <= 0.5):    # License should be between 5% and 50% of image
                    
                    # Draw detection for debugging
                    debug_detection = debug_original.copy()
                    cv2.drawContours(debug_detection, [box], 0, (0, 255, 0), 2)
                    cv2.imwrite(f"{prefix}debug_8_detection_{i}_{idx}.png", debug_detection)
                    
                    print(f"Found license using method {idx}")
                    print(f"Aspect ratio: {aspect_ratio:.2f}")
                    print(f"Area ratio: {(area/img_area):.3f}")
                    
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

def create_pdf(front_path, back_path, output_path, single_page=False):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Add front image
    img = ImageReader(front_path)
    c.drawImage(img, 50, height/2 - 100, width-100, 200, preserveAspectRatio=True)
    
    if not single_page:
        # Add back image only if dual-page mode
        img = ImageReader(back_path)
        c.drawImage(img, 50, height/4 - 100, width-100, 200, preserveAspectRatio=True)
    
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
