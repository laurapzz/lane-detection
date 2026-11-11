from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

from general.utils import check_data_folder
from lane_detection import lane_detection_image

# Global image path and ROI mask variables
image_path = None
current_roi_mask = None  # Stores binary mask for downstream obstacle detection


def extract_lane_points_from_csv(csv_path):
    """
    Parses lane points from your format where columns 'Ego_Left_Lane' 
    and 'Ego_Right_Lane' contain string lists of points: "x1,y1;x2,y2;..."
    """
    df = pd.read_csv(csv_path)

    # Grab the row matching the image (or the last row)
    row = df.iloc[-1]

    def parse_lane_string(lane_str):
        if pd.isna(lane_str) or not str(lane_str).strip():
            return np.empty((0, 2), dtype=np.int32)
        
        pts = []
        # Split string by ';' to get individual 'x,y' pairs
        for point_str in str(lane_str).split(';'):
            point_str = point_str.strip()
            if not point_str:
                continue
            parts = point_str.split(',')
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                pts.append([x, y])
        
        pts = np.array(pts, dtype=np.int32)
        
        # Sort from top of the image (smaller Y) to bottom of the image (larger Y)
        if len(pts) > 0:
            pts = pts[np.argsort(pts[:, 1])]
            
        return pts

    left_pts = parse_lane_string(row['Ego_Left_Lane'])
    right_pts = parse_lane_string(row['Ego_Right_Lane'])

    if len(left_pts) == 0 or len(right_pts) == 0:
        raise ValueError(f"Could not parse valid left or right lane points from: {csv_path}")

    return left_pts, right_pts


def generate_lane_roi(frame, left_pts, right_pts):
    """Generates a binary ROI mask and visual green/red overlay image."""
    height, width = frame.shape[:2]

    if len(left_pts) == 0 or len(right_pts) == 0:
        raise ValueError("Lane points could not be extracted from CSV.")

    # Bottom endpoints of the detected lines
    bottom_left_x = left_pts[-1][0]
    bottom_right_x = right_pts[-1][0]

    # Extend straight down to the bottom of the frame
    ext_bottom_left = [bottom_left_x, height - 1]
    ext_bottom_right = [bottom_right_x, height - 1]

    # Construct continuous boundary polygon:
    # Left line (top->bottom) -> Left extension down -> Right extension down -> Right line (bottom->top)
    polygon_pts = np.vstack([
        left_pts,
        [ext_bottom_left],
        [ext_bottom_right],
        right_pts[::-1]
    ]).astype(np.int32)

    # Create binary mask (255 inside ROI, 0 outside)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)

    # Create visual overlay matching reference image
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_pts], (0, 200, 100))  # Green ROI fill
    visual_result = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Draw red highlight boundary
    cv2.polylines(visual_result, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return mask, visual_result, polygon_pts


def select_image():
    global image_path
    image_path = None
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        image_path = file_path
        messagebox.showinfo("Image Selected", f"Selected Image: {image_path}")
    else:
        messagebox.showwarning("No Selection", "No image selected.")


def perform_lane_detection():
    global image_path, current_roi_mask

    if not image_path:
        messagebox.showwarning("No Image", "Please select an image first.")
        return

    try:
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("Failed to read image.")

        frame_height, frame_width = frame.shape[:2]

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight() // 1.5
        scaling_factor = min(screen_width / frame_width, screen_height / frame_height)
        new_size = (int(frame_width * scaling_factor), int(frame_height * scaling_factor))
        resized_frame = cv2.resize(frame, new_size)

        display_window = tk.Toplevel(root)
        display_window.title("Lane Detection - Select Cutoff Line")

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

        def update_line(val):
            y_pos = int(val) * scaling_factor
            canvas.delete("line")
            canvas.create_line(0, y_pos, new_size[0], y_pos, fill="red", tags="line")

        def accept():
            global current_roi_mask

            y_pos = int(slider.get())
            display_window.destroy()

            # 1. Execute detection model yielding output frame and saved CSV
            annotated_frame, output_csv = lane_detection_image(image_path, LINE_POINT=y_pos)

            # 2. Extract lane points from generated CSV
            left_pts, right_pts = extract_lane_points_from_csv(output_csv)

            # 3. Generate ROI Mask and Visual Overlay
            current_roi_mask, visual_result, poly_pts = generate_lane_roi(frame, left_pts, right_pts)

            # 4. Save ROI mask for obstacle detection use
            mask_output_path = str(Path(output_csv).parent / "lane_roi_mask.png")
            cv2.imwrite(mask_output_path, current_roi_mask)

            # 5. Show Preview Windows
            preview_overlay = cv2.resize(visual_result, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            preview_mask = cv2.resize(current_roi_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.imshow("Lane Detection & Obstacle ROI Overlay", preview_overlay)
            cv2.imshow("Binary Obstacle ROI Mask", preview_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            messagebox.showinfo(
                "Lane Detection",
                f"Detection completed successfully.\nCSV saved to: {output_csv}\nROI Mask saved to: {mask_output_path}"
            )

        slider = tk.Scale(display_window, from_=0, to=frame_height, orient=tk.VERTICAL, command=update_line)
        slider.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(display_window, width=new_size[0], height=new_size[1])
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        accept_button = tk.Button(display_window, text="Accept & Run", command=accept)
        accept_button.pack(pady=20)

        canvas.image = imgtk
        display_window.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Main Window GUI
root = tk.Tk()
root.title("Single Image Lane Detection System")
root.geometry("350x200")

select_button = tk.Button(root, text="Select Image", command=select_image, width=25)
select_button.pack(pady=20)

lane_detect_button = tk.Button(root, text="Perform Lane Detection (2 Lines)", command=perform_lane_detection, width=25)
lane_detect_button.pack(pady=20)

root.mainloop() 
