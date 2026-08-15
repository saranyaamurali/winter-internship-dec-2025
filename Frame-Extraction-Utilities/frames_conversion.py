import cv2
import os
import zipfile
import shutil

input_path = "windmill videos"  
output_folder = "windmill extracted"
frames_per_second = 1

os.makedirs(output_folder, exist_ok=True)

video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.bin']
temp_extract_folder = "temp_extracted"

def extract_frames(video_path, output_path, video_name):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"  ERROR: Could not open video")
        return 0
    
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_gap = int(video_fps / frames_per_second) if frames_per_second > 0 else 1
    os.makedirs(output_path, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count % frame_gap == 0:
            frame_filename = f"{video_name}_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_path, frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    video.release()
    return saved_count

# Check if input_path is a zip file
if os.path.isfile(input_path + ".zip"):
    input_path = input_path + ".zip"

if not os.path.exists(input_path):
    print(f"ERROR: '{input_path}' not found!")
    exit()

# If input is a zip file, extract it first
if os.path.isfile(input_path) and input_path.lower().endswith('.zip'):
    print(f"Input is a zip file: {input_path}")
    print(f"Extracting...")
    
    try:
        os.makedirs(temp_extract_folder, exist_ok=True)
        with zipfile.ZipFile(input_path, 'r') as z:
            z.extractall(temp_extract_folder)
        print(f"✓ Extracted to {temp_extract_folder}\n")
        input_folder = temp_extract_folder
    except Exception as e:
        print(f"ERROR: Could not extract zip: {e}")
        exit()
else:
    input_folder = input_path

# Find all video files and zip files in the folder
all_videos = []
all_zips = []

for root, dirs, files in os.walk(input_folder):
    for f in files:
        if any(f.lower().endswith(fmt) for fmt in video_formats):
            all_videos.append(os.path.join(root, f))
        elif f.lower().endswith('.zip'):
            all_zips.append(os.path.join(root, f))

if not all_videos and not all_zips:
    print(f"ERROR: No video or zip files found!")
    if os.path.exists(temp_extract_folder):
        shutil.rmtree(temp_extract_folder)
    exit()

print(f"Found {len(all_videos)} video(s) and {len(all_zips)} zip file(s)\n")

file_index = 0
total_files = len(all_videos) + len(all_zips)

# Process video files
for video_path in all_videos:
    file_index += 1
    video_file = os.path.basename(video_path)
    print(f"[{file_index}/{total_files}] Processing: {video_file}")
    
    video_name = os.path.splitext(video_file)[0]
    video_output = os.path.join(output_folder, video_name + "_frames")
    
    saved_count = extract_frames(video_path, video_output, video_name)
    if saved_count > 0:
        print(f"  ✓ Extracted {saved_count} frames → {video_output}\n")

# Process zip files inside the folder
for zip_path in all_zips:
    file_index += 1
    zip_file = os.path.basename(zip_path)
    print(f"[{file_index}/{total_files}] Processing: {zip_file}")
    
    zip_name = os.path.splitext(zip_file)[0]
    extract_path = os.path.join(temp_extract_folder, zip_name)
    
    try:
        print(f"  Extracting {zip_file}...")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
        print(f"  ✓ Extracted")
        
        # Find all videos in extracted folder
        videos = []
        for root, dirs, files in os.walk(extract_path):
            for f in files:
                if any(f.lower().endswith(fmt) for fmt in video_formats):
                    videos.append(os.path.join(root, f))
        
        if not videos:
            print(f"  ⚠ No videos found in zip\n")
            continue
        
        print(f"  Found {len(videos)} video(s)")
        
        for video_path in videos:
            vname = os.path.splitext(os.path.basename(video_path))[0]
            print(f"    Processing: {vname}")
            
            video_output = os.path.join(output_folder, f"{vname}_frames")
            saved_count = extract_frames(video_path, video_output, vname)
            
            if saved_count > 0:
                print(f"      ✓ Extracted {saved_count} frames → {video_output}")
        
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}\n")

# Clean up temporary extracted folder
if os.path.exists(temp_extract_folder):
    shutil.rmtree(temp_extract_folder)
    print("Cleaned up temporary files")

print("\nALL VIDEOS PROCESSED!")