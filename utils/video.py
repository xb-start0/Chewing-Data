import os
import subprocess
from pathlib import Path
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def find_mouth_and_nose_bounds(image, padding_ratio=0.2):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(p.x * image.shape[1]), int(p.y * image.shape[0])) for p in landmarks]


    mouth_points = [points[i] for i in range(61, 96)]  
    mouth_center_x = sum(p[0] for p in mouth_points) // len(mouth_points)
    mouth_center_y = sum(p[1] for p in mouth_points) // len(mouth_points)


    nose_points = [points[48]]
    if nose_points:
        min_nose_y = min(nose_points, key=lambda p: p[1])[1]
        adjusted_min_nose_y = min_nose_y

    else:
        adjusted_min_nose_y = int(image.shape[0] * 0.4)  


    chin_points = [points[i] for i in range(172, 198)]  
    if chin_points:
        max_chin_y = max(chin_points, key=lambda p: p[1])[1]
    else:
        max_chin_y = int(image.shape[0] * 0.8)  


    height = max_chin_y - adjusted_min_nose_y + int((max_chin_y - adjusted_min_nose_y) * padding_ratio)


    width = int(height * 1.5) 
    half_width = width // 2

  
    offset = int(width * 0.15) 
    crop_x = max(0, mouth_center_x - half_width + offset) 
    crop_y = max(0, adjusted_min_nose_y) 


    crop_width = min(width, image.shape[1] - crop_x)
    crop_height = min(height, image.shape[0] - crop_y)

    return crop_width, crop_height, crop_x, crop_y

def crop_video_based_on_face(input_file, output_file, overwrite=False, padding_ratio=0.2):
    if os.path.exists(output_file) and not overwrite:
        print(f"文件已存在且未设置覆盖: {output_file}")
        return

    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()
    if not ret:
        print(f"无法读取视频文件: {input_file}")
        cap.release()
        return

    bounds = find_mouth_and_nose_bounds(frame, padding_ratio)
    if bounds is None:
        print(f"未找到脸部或嘴巴: {input_file}")
        cap.release()
        return

    crop_width, crop_height, crop_x, crop_y = bounds

    command = [
        'ffmpeg', '-i', input_file,
        '-vf', f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
        '-r', '30',
        '-c:a', 'copy',
        '-y' if overwrite else '-n', 
        output_file
    ]
    print(command)
    try:
        subprocess.run(command, check=True)
        print(f"成功裁剪: {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"裁剪失败: {input_file}, 错误信息: {e}")

    cap.release()



def process_videos_in_nested_dirs(root_dir,out_dir,depth=3, overwrite=False, padding_ratio=0.2):
    for dir in os.listdir(root_dir):
        # input_file_audio = root + dir_name + "/bread/"+ dir_name+ "_bread_bone.wav"
        input_file_video = os.path.join(root_dir,dir,f"bread/{dir}_bread_front.mp4")
        out_dir_final = os.path.join(out_dir,dir)
        # input_file_video = root + dir_name + '/'
        if not os.path.exists(out_dir_final):
            os.makedirs(out_dir_final,exist_ok=True)
        save_file_video = os.path.join(out_dir_final,f"{dir}_bread_front.mp4")
        crop_video_based_on_face(input_file_video, save_file_video, overwrite=overwrite, padding_ratio=padding_ratio)
            

root_directory = ''
out_directory = ''
overwrite_existing_files = True 
padding_ratio = 0.25  

# 调用函数开始处理
process_videos_in_nested_dirs(root_directory, out_directory,depth=3, overwrite=overwrite_existing_files, padding_ratio=padding_ratio)
face_mesh.close()
