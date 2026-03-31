import gradio as gr
import cv2
import numpy as np
from PIL import Image

def adjust_exposure(image, gamma):
    """Adjust exposure using gamma correction"""
    img_array = np.array(image)
    img_norm = img_array.astype(np.float32) / 255.0
    adjusted = np.power(img_norm, 1/gamma)
    return np.clip(adjusted * 255, 0, 255).astype(np.uint8)

def compute_weight_maps(images):
    """Compute weight maps for fusion"""
    weights = []
    
    for img in images:
        # Contrast
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = np.abs(contrast)
        contrast = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)
        
        # Saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = hsv[:,:,1].astype(np.float32) / 255.0
        
        # Well-exposedness
        img_float = img.astype(np.float32) / 255.0
        well_exposed = 1.0 - np.abs(img_float - 0.5) * 2
        well_exposed = np.prod(well_exposed, axis=2)
        
        # Combine
        weight = contrast * saturation * well_exposed
        weight = cv2.normalize(weight, None, 0, 1, cv2.NORM_MINMAX)
        weights.append(weight)
    
    # Normalize
    weight_sum = np.sum(weights, axis=0) + 1e-6
    weights = [w / weight_sum for w in weights]
    
    return weights

def fuse_images(images, weights):
    """Fuse multiple exposures"""
    fused = np.zeros_like(images[0].astype(np.float32))
    
    for img, weight in zip(images, weights):
        weight_3ch = np.stack([weight] * 3, axis=2)
        fused += img.astype(np.float32) * weight_3ch
    
    return np.clip(fused, 0, 255).astype(np.uint8)

def post_process(image, clahe_clip=2.0):
    """Apply post-processing"""
    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

def enhance_image(image, gamma, clahe_clip, denoise_strength, method):
    """
    Enhance a low-light image
    
    Parameters:
    - image: Input image
    - gamma: Gamma correction value
    - clahe_clip: CLAHE clip limit
    - denoise_strength: Denoising strength
    - method: Enhancement method
    """
    if image is None:
        return None, "Please upload an image"
    
    try:
        img_array = np.array(image)
        
        if method == "Multi-Frame Fusion":
            # Multi-exposure fusion
            exposure_levels = [0.5, 1.0, 1.5, 2.0]
            exposures = []
            
            for gamma_val in exposure_levels:
                adjusted = adjust_exposure(image, gamma_val)
                exposures.append(adjusted)
            
            weights = compute_weight_maps(exposures)
            fused = fuse_images(exposures, weights)
            
            # Post-process
            enhanced = post_process(fused, clahe_clip)
            
            status = "✅ Multi-frame fusion applied"
            
        elif method == "Gamma + CLAHE":
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
            
            # Denoising
            if denoise_strength > 0:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced, None, denoise_strength, denoise_strength, 7, 21
                )
            
            status = f"✅ Gamma: {gamma}, CLAHE: {clahe_clip} applied"
        
        elif method == "CLAHE Only":
            # CLAHE only
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            if denoise_strength > 0:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced, None, denoise_strength, denoise_strength, 7, 21
                )
            
            status = f"✅ CLAHE: {clahe_clip} applied"
        
        else:  # Gamma Only
            # Gamma correction only
            if gamma != 1.0:
                invGamma = 1.0 / gamma
                img_array = np.power(img_array / 255.0, invGamma)
                img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            
            if denoise_strength > 0:
                img_array = cv2.fastNlMeansDenoisingColored(
                    img_array, None, denoise_strength, denoise_strength, 7, 21
                )
            
            enhanced = img_array
            status = f"✅ Gamma: {gamma} applied"
        
        return enhanced, status
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def create_comparison(original, enhanced):
    """Create side-by-side comparison"""
    if original is None or enhanced is None:
        return None
    
    # Convert to same size
    orig_array = np.array(original)
    enh_array = np.array(enhanced)
    
    # Resize to match if needed
    if orig_array.shape != enh_array.shape:
        enh_array = cv2.resize(enh_array, (orig_array.shape[1], orig_array.shape[0]))
    
    # Create side-by-side
    comparison = np.hstack([orig_array, enh_array])
    
    return comparison

# Create Gradio interface with advanced options
with gr.Blocks(title="📸 Low-Light Enhancement", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 📸 Smart Low-Light Image Enhancement
    ### Professional-grade enhancement for mobile camera images
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Low-Light Image", height=300)
            
            with gr.Accordion("⚙️ Enhancement Settings", open=True):
                method = gr.Radio(
                    ["Multi-Frame Fusion", "Gamma + CLAHE", "CLAHE Only", "Gamma Only"],
                    label="Enhancement Method",
                    value="Multi-Frame Fusion",
                    info="Choose the enhancement algorithm"
                )
                
                gamma = gr.Slider(
                    minimum=0.3,
                    maximum=2.5,
                    value=1.0,
                    step=0.1,
                    label="Gamma Correction",
                    info="Adjust brightness (lower = brighter)"
                )
                
                clahe_clip = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                    label="CLAHE Clip Limit",
                    info="Contrast enhancement strength"
                )
                
                denoise_strength = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Denoise Strength",
                    info="Noise reduction (0 = off)"
                )
            
            enhance_btn = gr.Button("🚀 Enhance Image", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="📥 Enhanced Result", height=300)
            status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Accordion("📊 Comparison View", open=False):
                comparison_image = gr.Image(type="pil", label="Before/After Side-by-Side")
    
    # Examples
    gr.Markdown("### 📚 How to Use")
    gr.Markdown("""
    1. **Upload** a low-light image using the upload button
    2. **Select** your preferred enhancement method:
       - **Multi-Frame Fusion**: Best quality, combines multiple exposures
       - **Gamma + CLAHE**: Balanced enhancement with contrast adjustment
       - **CLAHE Only**: Fast contrast enhancement
       - **Gamma Only**: Simple brightness adjustment
    3. **Adjust** the settings sliders as needed
    4. **Click** "Enhance Image" to process
    5. **Download** your enhanced image
    """)
    
    gr.Markdown("""
    ### 💡 Tips
    - **Multi-Frame Fusion** works best for very dark images
    - Lower **Gamma** values (0.5-0.8) brighten dark images
    - Higher **CLAHE** values (2.0-3.0) increase contrast
    - Use **Denoise** to reduce noise in dark areas
    """)
    
    # Event handlers
    enhance_btn.click(
        fn=enhance_image,
        inputs=[input_image, gamma, clahe_clip, denoise_strength, method],
        outputs=[output_image, status_text]
    ).then(
        fn=create_comparison,
        inputs=[input_image, output_image],
        outputs=[comparison_image]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **Made with ❤️ for Low-Light Image Enhancement** | Powered by Gradio
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)