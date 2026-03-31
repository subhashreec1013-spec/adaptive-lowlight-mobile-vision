import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class LowLightEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Low-Light Image Enhancement App")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1e1e1e')
        
        # Set dark theme colors
        self.bg_color = '#1e1e1e'
        self.fg_color = '#ffffff'
        self.accent_color = '#0078d4'
        
        self.original_image = None
        self.enhanced_image = None
        self.image_path = None
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('Accent.TButton', background=self.accent_color, foreground='white')
        style.configure('TLabelframe', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 11, 'bold'))
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ========== LEFT PANEL - Image Display ==========
        left_frame = ttk.LabelFrame(main_frame, text="📷 Image Preview", padding="10")
        left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display area
        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_frame, text="No image loaded\n\nClick 'Load Image' to begin", 
                                     background='#2d2d2d', foreground='#888888',
                                     font=('Segoe UI', 14))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ========== RIGHT PANEL - Controls ==========
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(right_frame, text="⚙️ Enhancement Controls", 
                               font=('Segoe UI', 14, 'bold'))
        title_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Load Image Button
        load_btn = ttk.Button(right_frame, text="📂 Load Image", command=self.load_image, width=25)
        load_btn.pack(fill=tk.X, pady=5)
        
        # Enhancement Method
        method_frame = ttk.LabelFrame(right_frame, text="Enhancement Method", padding="10")
        method_frame.pack(fill=tk.X, pady=(20, 10))
        
        self.method_var = tk.StringVar(value="Multi-Frame Fusion")
        methods = ["Multi-Frame Fusion", "Gamma + CLAHE", "CLAHE Only", "Gamma Only"]
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, values=methods, state="readonly")
        method_combo.pack(fill=tk.X, pady=5)
        
        # Gamma Correction
        gamma_frame = ttk.LabelFrame(right_frame, text="Gamma Correction", padding="10")
        gamma_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = ttk.Scale(gamma_frame, from_=0.3, to=2.5, variable=self.gamma_var, orient=tk.HORIZONTAL)
        gamma_scale.pack(fill=tk.X, pady=5)
        self.gamma_label = ttk.Label(gamma_frame, text="Value: 1.0")
        self.gamma_label.pack()
        gamma_scale.config(command=lambda v: self.gamma_label.config(text=f"Value: {float(v):.2f}"))
        
        # CLAHE Clip Limit
        clahe_frame = ttk.LabelFrame(right_frame, text="CLAHE Clip Limit", padding="10")
        clahe_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.clahe_var = tk.DoubleVar(value=2.0)
        clahe_scale = ttk.Scale(clahe_frame, from_=1.0, to=5.0, variable=self.clahe_var, orient=tk.HORIZONTAL)
        clahe_scale.pack(fill=tk.X, pady=5)
        self.clahe_label = ttk.Label(clahe_frame, text="Value: 2.0")
        self.clahe_label.pack()
        clahe_scale.config(command=lambda v: self.clahe_label.config(text=f"Value: {float(v):.2f}"))
        
        # Denoise Strength
        denoise_frame = ttk.LabelFrame(right_frame, text="Denoise Strength", padding="10")
        denoise_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.denoise_var = tk.IntVar(value=10)
        denoise_scale = ttk.Scale(denoise_frame, from_=0, to=30, variable=self.denoise_var, orient=tk.HORIZONTAL)
        denoise_scale.pack(fill=tk.X, pady=5)
        self.denoise_label = ttk.Label(denoise_frame, text="Value: 10")
        self.denoise_label.pack()
        denoise_scale.config(command=lambda v: self.denoise_label.config(text=f"Value: {int(float(v))}"))
        
        # Enhance Button
        enhance_btn = ttk.Button(right_frame, text="🚀 Enhance Image", command=self.enhance_image, 
                                style='Accent.TButton')
        enhance_btn.pack(fill=tk.X, pady=20)
        
        # Save Button
        save_btn = ttk.Button(right_frame, text="💾 Save Result", command=self.save_image)
        save_btn.pack(fill=tk.X, pady=5)
        
        # Reset Button
        reset_btn = ttk.Button(right_frame, text="🔄 Reset", command=self.reset_app)
        reset_btn.pack(fill=tk.X, pady=5)
        
        # ========== BOTTOM PANEL - Comparison ==========
        bottom_frame = ttk.LabelFrame(main_frame, text="📊 Before/After Comparison", padding="10")
        bottom_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create two columns for before/after
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        
        self.before_frame = ttk.Frame(bottom_frame)
        self.before_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.before_label = ttk.Label(self.before_frame, text="Before", background='#2d2d2d', 
                                      foreground='#888888', font=('Segoe UI', 12))
        self.before_label.pack(fill=tk.BOTH, expand=True)
        
        self.after_frame = ttk.Frame(bottom_frame)
        self.after_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.after_label = ttk.Label(self.after_frame, text="After", background='#2d2d2d', 
                                     foreground='#888888', font=('Segoe UI', 12))
        self.after_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path)
                
                # Convert to RGB if necessary
                if self.original_image.mode != 'RGB':
                    self.original_image = self.original_image.convert('RGB')
                
                # Display in main area
                self.display_image(self.original_image, self.image_label, size=(500, 400))
                
                # Display in before/after
                self.display_image(self.original_image, self.before_label, size=(300, 200))
                
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.enhanced_image = None
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def display_image(self, img, label, size=None):
        if size:
            img_display = img.copy()
            img_display.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            img_display = img
        
        photo = ImageTk.PhotoImage(img_display)
        label.config(image=photo, text="")
        label.image = photo
    
    def enhance_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        img_array = np.array(self.original_image)
        gamma = self.gamma_var.get()
        clahe_clip = self.clahe_var.get()
        denoise = self.denoise_var.get()
        method = self.method_var.get()
        
        self.status_var.set("Processing... Please wait...")
        self.root.update()
        
        try:
            if method == "Multi-Frame Fusion":
                enhanced_array = self.multi_frame_fusion(img_array)
                
            elif method == "Gamma + CLAHE":
                enhanced_array = self.gamma_clahe_enhancement(img_array, gamma, clahe_clip, denoise)
            
            elif method == "CLAHE Only":
                enhanced_array = self.clahe_only(img_array, clahe_clip, denoise)
            
            else:  # Gamma Only
                enhanced_array = self.gamma_only(img_array, gamma, denoise)
            
            self.enhanced_image = Image.fromarray(enhanced_array)
            self.display_image(self.enhanced_image, self.image_label, size=(500, 400))
            self.display_image(self.enhanced_image, self.after_label, size=(300, 200))
            
            self.status_var.set("✅ Enhancement complete!")
            messagebox.showinfo("Success", "Image enhanced successfully!")
            
        except Exception as e:
            self.status_var.set("❌ Error occurred")
            messagebox.showerror("Error", f"Enhancement failed:\n{str(e)}")
    
    def multi_frame_fusion(self, img_array):
        """Multi-exposure fusion enhancement"""
        exposure_levels = [0.5, 1.0, 1.5, 2.0]
        exposures = []
        
        for gamma in exposure_levels:
            img_norm = img_array.astype(np.float32) / 255.0
            adjusted = np.power(img_norm, 1/gamma)
            adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
            exposures.append(adjusted)
        
        # Compute weight maps
        weights = []
        for img in exposures:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            contrast = cv2.Laplacian(gray, cv2.CV_32F)
            contrast = np.abs(contrast)
            contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1].astype(np.float32) / 255.0
            
            img_float = img.astype(np.float32) / 255.0
            well_exposed = 1.0 - np.abs(img_float - 0.5) * 2
            well_exposed = np.prod(well_exposed, axis=2)
            
            weight = contrast * saturation * well_exposed
            weight = cv2.normalize(weight, None, 0, 1, cv2.NORM_MINMAX)
            weights.append(weight)
        
        # Normalize weights
        weight_sum = np.sum(weights, axis=0) + 1e-6
        weights = [w / weight_sum for w in weights]
        
        # Fuse
        fused = np.zeros_like(img_array.astype(np.float32))
        for img, weight in zip(exposures, weights):
            weight_3ch = np.stack([weight] * 3, axis=2)
            fused += img.astype(np.float32) * weight_3ch
        
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        
        # Post-process with CLAHE
        lab = cv2.cvtColor(fused, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Denoise
        if self.denoise_var.get() > 0:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def gamma_clahe_enhancement(self, img_array, gamma, clahe_clip, denoise):
        """Gamma correction + CLAHE enhancement"""
        # Gamma correction
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            img_array = np.power(img_array / 255.0, invGamma)
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        
        # CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Denoise
        if denoise > 0:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def clahe_only(self, img_array, clahe_clip, denoise):
        """CLAHE only enhancement"""
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        if denoise > 0:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def gamma_only(self, img_array, gamma, denoise):
        """Gamma correction only"""
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            img_array = np.power(img_array / 255.0, invGamma)
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        
        if denoise > 0:
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        return img_array
    
    def save_image(self):
        if self.enhanced_image is None:
            messagebox.showwarning("Warning", "No enhanced image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.enhanced_image.save(file_path)
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def reset_app(self):
        self.original_image = None
        self.enhanced_image = None
        self.image_path = None
        
        self.image_label.config(image="", text="No image loaded\n\nClick 'Load Image' to begin")
        self.before_label.config(image="", text="Before")
        self.after_label.config(image="", text="After")
        
        self.gamma_var.set(1.0)
        self.clahe_var.set(2.0)
        self.denoise_var.set(10)
        self.method_var.set("Multi-Frame Fusion")
        
        self.gamma_label.config(text="Value: 1.0")
        self.clahe_label.config(text="Value: 2.0")
        self.denoise_label.config(text="Value: 10")
        
        self.status_var.set("Ready")

def main():
    root = tk.Tk()
    app = LowLightEnhancementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()