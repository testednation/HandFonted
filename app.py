import gradio as gr
import os
import shutil
import tempfile
import torch
from paddleocr import TextDetection

from character_classification import classify_characters, ResInceptionNet
from character_segmentation import segment_characters
from font_creation import update_font_from_images

# --- Load Models Once ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

CLASSIFICATION_MODEL_PATH = "resources/best_ResInceptionNet_model0.8811.pth"
CLASSIFICATION_MODEL = ResInceptionNet(num_classes=52)
CLASSIFICATION_MODEL.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=torch.device('cpu')))
CLASSIFICATION_MODEL.eval()

DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
DETECTION_MODEL_PATH = r"resources/PP-OCRv5_mobile_det_infer"
DETECTION_MODEL = TextDetection(model_name=DETECTION_MODEL_NAME, model_dir=DETECTION_MODEL_PATH, enable_mkldnn=False)

BASE_FONT_PATH = "resources/Arimo-Regular.ttf"


def generate_font(input_images, font_name, font_style, thickness, is_freeform, confidence_threshold):
    """
    Gradio backend function to process uploaded images and generate the font.
    """
    if not input_images:
        return "Please upload at least one image.", None

    try:
        all_char_images = []

        # 1. Segment characters from all input images
        for img_path in input_images:
             # Gradio provides file paths
             chars = segment_characters(img_path, det_model=DETECTION_MODEL)
             if chars:
                 all_char_images.extend(chars)

        if not all_char_images:
            return "❌ Error: No characters were segmented from the image(s). Please try clearer images.", None

        # 2. Classify Characters
        char_images_with_labels = classify_characters(
            all_char_images,
            model=CLASSIFICATION_MODEL,
            OUTPUT_PATH="",
            use_global_assignment=not is_freeform,
            confidence_threshold=confidence_threshold
        )

        if not char_images_with_labels:
            return "❌ Error: Character classification failed or no characters met the confidence threshold.", None

        # 3. Create Font in a temporary directory
        output_dir = tempfile.mkdtemp()
        output_font_path = os.path.join(output_dir, f"{font_name.replace(' ', '_')}.ttf")

        update_font_from_images(
            font_path=BASE_FONT_PATH,
            char_image_list=char_images_with_labels,
            output_path=output_font_path,
            desired_thickness=thickness,
            new_family_name=font_name,
            new_style_name=font_style
        )

        success_msg = f"🎉 Success! Generated font with {len(char_images_with_labels)} characters."
        return success_msg, output_font_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ An error occurred during processing: {str(e)}", None

# --- Gradio UI Layout ---
with gr.Blocks(title="HandFonted Web UI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ✍️ HandFonted Interactive UI")
    gr.Markdown("Turn your handwriting into a functional `.ttf` font file! Upload photos of your handwriting below.")

    with gr.Row():
        with gr.Column(scale=1):
            input_images = gr.File(label="Upload Handwriting Images", file_count="multiple", type="filepath", file_types=["image"])

            with gr.Group():
                gr.Markdown("### Font Settings")
                font_name = gr.Textbox(label="Font Name", value="My Handwriting")
                font_style = gr.Textbox(label="Font Style", value="Regular")
                thickness = gr.Slider(label="Stroke Thickness", minimum=10, maximum=300, value=100, step=5)

            with gr.Group():
                gr.Markdown("### Processing Mode")
                is_freeform = gr.Checkbox(label="Freeform Notes Mode", value=True, info="Check this if uploading random notes. Uncheck if uploading the strict A-Z template.")
                conf_thresh = gr.Slider(label="Confidence Threshold (Freeform Mode)", minimum=0.1, maximum=1.0, value=0.5, step=0.05)

            generate_btn = gr.Button("Generate Font", variant="primary")

        with gr.Column(scale=1):
            output_status = gr.Textbox(label="Status", interactive=False)
            output_file = gr.File(label="Download Generated Font", interactive=False)

    generate_btn.click(
        fn=generate_font,
        inputs=[input_images, font_name, font_style, thickness, is_freeform, conf_thresh],
        outputs=[output_status, output_file]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
