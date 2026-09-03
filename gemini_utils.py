import base64
import os
import datetime
import time
import random
import ssl
from google import genai
from google.genai import types
import httpx


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()

API_KEYS = [os.environ.get("API_KEY")]
MODEL = os.environ.get("MODEL")

VALID_API_KEYS = [key for key in API_KEYS if key is not None]

def generate(prompt, img_path=None, img_save_name=None, max_retries=15, base_delay=2, img_save_dir="./apple_images_counterfactual", model=MODEL):
    if not VALID_API_KEYS:
        raise ValueError("No valid API keys found in environment variables")
        
    retry_count = 0
    final_text_responses = []
    final_generated_images = []

    while retry_count <= max_retries:
        selected_api_key = random.choice(VALID_API_KEYS)
        api_key_display = f"...{selected_api_key[-6:]}" if len(selected_api_key) > 6 else selected_api_key
        
        client = genai.Client(
            api_key=selected_api_key,
        )

        try:
            uploaded_files_for_attempt = []
            if img_path:
                for img_file_path in img_path:
                    if not os.path.exists(img_file_path):
                        print(f"Warning: Image file not found: {img_file_path}")
                        return None
                    try:
                        uploaded_files_for_attempt.append(
                            client.files.upload(file=img_file_path)
                        )
                    except Exception as e_upload:
                        print(f"Error during file upload {img_file_path} with key {api_key_display}: {e_upload}. This attempt will fail, retrying main operation.")
                        raise e_upload

                parts = []
                for file_obj in uploaded_files_for_attempt:
                    parts.append(
                        types.Part.from_uri(
                            file_uri=file_obj.uri,
                            mime_type=file_obj.mime_type,
                        )
                    )
                
                parts.append(types.Part.from_text(text=prompt))
                
                contents = [
                    types.Content(
                        role="user",
                        parts=parts,
                    ),
                ]
            else:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                        ],
                    ),
                ]

            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_modalities=[
                    "image",
                    "text",
                ],
                response_mime_type="text/plain",
            )

            attempt_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            current_text_responses = []
            current_generated_images = []
            image_counter_this_attempt = 0
            
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                if chunk.candidates[0].content.parts[0].inline_data:
                    mime_type = chunk.candidates[0].content.parts[0].inline_data.mime_type
                    extension = mime_type.split('/')[-1]
                    image_counter_this_attempt += 1
                    
                    file_name_to_save: str
                    if img_save_name:
                        if os.path.isabs(img_save_name) or os.path.dirname(img_save_name):
                            file_name_to_save = img_save_name
                        else:
                            file_name_to_save = os.path.join(img_save_dir, img_save_name)
                        if image_counter_this_attempt > 1 and img_save_name:
                             name, ext = os.path.splitext(file_name_to_save)
                             file_name_to_save = f"{name}_{image_counter_this_attempt}{ext}"

                    else:
                        file_name_to_save = os.path.join(img_save_dir, f"gemini_output_{attempt_timestamp}_{image_counter_this_attempt}.{extension}")

                    os.makedirs(os.path.dirname(file_name_to_save), exist_ok=True)
                    
                    save_binary_file(
                        file_name_to_save, chunk.candidates[0].content.parts[0].inline_data.data
                    )
                    current_generated_images.append(file_name_to_save)
                    print(
                        f"File of mime type {mime_type} saved to: {file_name_to_save}"
                    )
                else:
                    current_text_responses.append(chunk.text)
            
            final_text_responses = current_text_responses
            final_generated_images = current_generated_images
            break
            
        except (genai.errors.ClientError, genai.errors.ServerError, ssl.SSLCertVerificationError, httpx.ConnectError) as e:
            err_str = str(e)
            retry_count += 1
            
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached: {e}")
                raise

            delay_val = 10
            
            time.sleep(delay_val)
    
    result = {
        "text": "".join(final_text_responses) if final_text_responses else "",
        "images": final_generated_images
    }
    return result
