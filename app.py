import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import pandas as pd
import numpy as np

from general.utils import check_data_folder
from lane_detection import lane_detection_image

# Global image path variable
image_path = None

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
    global image_path

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
            y_pos = int(slider.get())
            display_window.destroy()

            # Execute single-image detection yielding 2 lines
            annotated_frame, output_csv = lane_detection_image(image_path, LINE_POINT=y_pos)

            preview = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow("Lane Detection Result (2 Ego Lines)", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            messagebox.showinfo("Lane Detection", f"Detection completed successfully.\nOutput saved to: {output_csv}")

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
