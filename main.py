import cv2
import json
import base64
import numpy as np
import os
import glob
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageEnhance
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()

def enhance_image(image):
    """Apply standard image enhancement"""
    enhanced_image = image.copy()
    
    # PIL enhancements
    pil_image = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    
    # Standard enhancement values
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.3)
    
    enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # CLAHE for shadows
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def pyramid_detection_system(image, overlap=0.1):
    """Pyramid detection - the proven method"""
    height, width = image.shape[:2]
    all_sections = {}
    
    # Level 1: 2-way
    top_end = int(height * 0.6)
    bottom_start = int(height * 0.4)
    all_sections["L1_top"] = image[0:top_end, :]
    all_sections["L1_bottom"] = image[bottom_start:height, :]
    
    # Level 2: 4-way
    section_height = int(height * 0.3)
    overlap_pixels = int(height * overlap)
    
    splits_4way = {
        'L2_top': (0, section_height + overlap_pixels),
        'L2_upper_mid': (int(height * 0.25) - overlap_pixels, int(height * 0.55) + overlap_pixels),
        'L2_lower_mid': (int(height * 0.45) - overlap_pixels, int(height * 0.75) + overlap_pixels),
        'L2_bottom': (int(height * 0.7) - overlap_pixels, height)
    }
    
    for name, (y_start, y_end) in splits_4way.items():
        all_sections[name] = image[y_start:y_end, :]
    
    # Level 3: 2x3 grid
    grid_h, grid_w = height // 2, width // 3
    overlap_h, overlap_w = int(grid_h * overlap), int(grid_w * overlap)
    
    for row in range(2):
        for col in range(3):
            y_start = max(0, row * grid_h - overlap_h)
            y_end = min(height, (row + 1) * grid_h + overlap_h)
            x_start = max(0, col * grid_w - overlap_w)
            x_end = min(width, (col + 1) * grid_w + overlap_w)
            
            section_name = f"L3_grid_{row}_{col}"
            all_sections[section_name] = image[y_start:y_end, x_start:x_end]
    
    return all_sections

def process_with_roboflow(image_array, client, section_name):
    """Process image with Roboflow"""
    try:
        temp_path = f"temp_{section_name}.jpg"
        cv2.imwrite(temp_path, image_array)
        
        result = client.run_workflow(
            workspace_name="barecodedetect",
            workflow_id="detect-count-and-visualize-2",
            images={"image": temp_path},
            use_cache=True
        )
        
        os.remove(temp_path)
        
        data = result[0] if isinstance(result, list) else result
        
        # Decode annotated image with bounding boxes
        annotated_image = None
        if 'output_image' in data:
            image_bytes = base64.b64decode(data['output_image'])
            nparr = np.frombuffer(image_bytes, np.uint8)
            annotated_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        tire_count = data.get('count_objects', 0)
        
        return {
            'count': tire_count,
            'annotated_image': annotated_image,
            'status': 'success'
        }
    
    except Exception as e:
        print(f"Error processing {section_name}: {e}")
        return {'status': 'error', 'error': str(e), 'count': 0}

def process_single_image(image_path, output_base_dir):
    """Process a single image through the pyramid detection system"""
    
    print(f"Loading image...")
    
    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image")
        return None
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_base_dir, f"{image_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhance image
    enhanced_image = enhance_image(original_image)
    
    # Save original and enhanced images
    cv2.imwrite(os.path.join(output_dir, "original_image.jpg"), original_image)
    cv2.imwrite(os.path.join(output_dir, "enhanced_image.jpg"), enhanced_image)
    
    # Initialize Roboflow client with environment variable
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.getenv('ROBOFLOW_API_KEY')
    )
    
    # Create pyramid sections
    print("Analyzing tire sections...")
    sections = pyramid_detection_system(enhanced_image)
    
    # Process each section
    results = {}
    total_count = 0
    
    for section_name, section_image in sections.items():
        result = process_with_roboflow(section_image, client, section_name)
        
        if result['status'] == 'success':
            results[section_name] = result
            total_count += result['count']
            
            # Save section image with detections
            if result.get('annotated_image') is not None:
                section_filename = f"{section_name}_detected_{result['count']}_tires.jpg"
                section_path = os.path.join(output_dir, section_filename)
                cv2.imwrite(section_path, result['annotated_image'])
    
    # Create simple report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tire_count": total_count,
        "section_results": {
            name: result['count'] for name, result in results.items()
        }
    }
    
    # Save JSON report
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Simple console output
    print(f"\nTIRE DETECTION RESULTS:")
    print(f"----------------------")
    
    # Group by levels for cleaner output
    level1 = {k: v for k, v in results.items() if k.startswith('L1_')}
    level2 = {k: v for k, v in results.items() if k.startswith('L2_')}
    level3 = {k: v for k, v in results.items() if k.startswith('L3_')}
    
    if level1:
        print("Level 1 sections:")
        for name, result in level1.items():
            clean_name = name.replace('L1_', '').replace('_', ' ').title()
            print(f"  {clean_name}: {result['count']} tires")
    
    if level2:
        print("Level 2 sections:")
        for name, result in level2.items():
            clean_name = name.replace('L2_', '').replace('_', ' ').title()
            print(f"  {clean_name}: {result['count']} tires")
    
    if level3:
        print("Level 3 sections:")
        for name, result in level3.items():
            clean_name = name.replace('L3_', '').replace('_', ' ').title()
            print(f"  {clean_name}: {result['count']} tires")
    
    print(f"----------------------")
    print(f"TOTAL TIRES: {total_count}")
    
    return {
        'total_count': total_count,
        'output_dir': output_dir,
        'report_data': report_data
    }

def main():
    """Main processing function"""
    
    print("TireCount Pro - Processing tire image...")
    
    # Setup directories
    input_dir = "./input_images"
    output_dir = "./output"
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the first image in input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_file = None
    
    for extension in image_extensions:
        files = glob.glob(os.path.join(input_dir, extension))
        if files:
            image_file = files[0]  # Take the first image found
            break
    
    if not image_file:
        print(f"No images found in {input_dir}")
        print("Please add a JPG or PNG image to the input_images folder")
        return
    
    print(f"Processing: {os.path.basename(image_file)}")
    
    # Process the image
    try:
        result = process_single_image(image_file, output_dir)
        
        if result:
            print(f"\nResults saved to: {result['output_dir']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()