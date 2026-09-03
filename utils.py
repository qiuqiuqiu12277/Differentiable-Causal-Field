import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
import clip
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from tqdm import tqdm
from datetime import datetime
import json, collections
from causal_graph_utils import *
from copy import deepcopy
import gemini_utils
import re


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class Helper:
    def __init__(self, datetime_str=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), N_sample=200, generate=None, output_path=None, model='gemini', dataset='MAG9', top_n=5):
        self.N_sample = N_sample
        self.top_n = top_n
        self.datetime_str = datetime_str
        if output_path is None:
            self.output_path = f"./outputs/MAG9/Ours"
        else:
            self.output_path = output_path
        self.generate = generate
        self.model = model
        self.img_generate = gemini_utils.generate
        self.dataset = dataset


    def extract_clip_embeddings(self, meta):
        """
        Extract CLIP embeddings for both text and images from all samples
        """
        text_embeddings = []
        image_embeddings = []
        
        print("Extracting CLIP embeddings...")
        for i in tqdm(range(len(meta))):
            review = meta.iloc[i]['Review']
            text_input = clip.tokenize([review[:min(len(review), 77)]]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_input)
            text_embeddings.append(text_embedding.cpu().numpy()[0])
            
            image_path = meta.iloc[i]['ImagePath']
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_embedding = model.encode_image(image)
                image_embeddings.append(image_embedding.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                image_embeddings.append(np.zeros_like(text_embeddings[-1]))
        
        return np.array(text_embeddings), np.array(image_embeddings)

    def compute_distance_pairs(self, embeddings1, embeddings2=None):
        """
        Compute distances between embeddings and return pairs with large distances
        If embeddings2 is provided, compute cross-modal distances
        Otherwise compute within-modal distances
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
            
        similarity = cosine_similarity(embeddings1, embeddings2)
        
        distance = 1 - similarity
        
        pairs = []
        
        if embeddings2 is embeddings1:
            indices = np.triu_indices_from(distance, k=1)
            distances = distance[indices]
            
            sorted_indices = np.argsort(-distances)
            
            top_n = min(10, len(sorted_indices))
            for i in range(top_n):
                idx = sorted_indices[i]
                row = indices[0][idx]
                col = indices[1][idx]
                pairs.append((row, col, distances[idx]))
                
        else:
            distances = distance.flatten()
            sorted_indices = np.argsort(-distances)
            
            top_n = min(10, len(sorted_indices))
            for i in range(top_n):
                idx = sorted_indices[i]
                row = idx // distance.shape[1]
                col = idx % distance.shape[1]
                pairs.append((row, col, distances[idx]))
        
        return pairs

    def generate_contrastive_prompt(self, meta, pairs, mode="text-text", used_factors=None):
        """
        Generate a contrastive prompt based on sample pairs
        """
        prompt_parts = []
        image_path = []
        image_batch = []

        used_factors_hint = ""
        if used_factors is not None and used_factors:
            used_factors_hint = "\n - Avoid overlapping with **those existing factors:**\n\t- " + '\n\t- '.join(used_factors) + '\n'
        
        prompt_parts.append(f"# {mode.title()} Contrastive Analysis\n")
        prompt_parts.append("I'm showing you pairs of samples with significant differences.\n")
        
        for idx, (i, j, distance) in enumerate(pairs[:self.top_n]):
            prompt_parts.append(f"\n## Pair {idx+1} (Distance: {distance:.4f})\n")
            
            prompt_parts.append(f"### Sample A (ID: {i})\n")
            prompt_parts.append(f"- **Score**: {meta.iloc[i]['score']}\n")
            if mode in ["text-text", "image-text", "text-image"]:
                prompt_parts.append(f"- **Review**: {meta.iloc[i]['Review']}\n")
            
            if mode in ["image-image", "text-image", "image-text"]:
                image_path.append(meta.iloc[i]['ImagePath'])
                image_batch.append(meta.iloc[i])
                prompt_parts.append(f"- **Image**: [Image {i}] (The image is provided with this prompt)\n")
            
            prompt_parts.append(f"\n### Sample B (ID: {j})\n")
            prompt_parts.append(f"- **Score**: {meta.iloc[j]['score']}\n")
            if mode in ["text-text", "image-text", "text-image"]:
                prompt_parts.append(f"- **Review**: {meta.iloc[j]['Review']}\n")
            
            if mode in ["image-image", "text-image", "image-text"]:
                image_path.append(meta.iloc[j]['ImagePath'])
                image_batch.append(meta.iloc[j])
                prompt_parts.append(f"- **Image**: [Image {j}] (The image is provided with this prompt)\n")
        
        prompt_parts.append("\nBased on these contrastive pairs, analyze the underlying interactions among potential factors that lead to the observed differences.\n")
        
        if self.dataset == 'MAG9':
            role_str = "You are an expert food analyst specializing in apple evaluation."
        elif self.dataset == 'Lung4':
            role_str = "You are an expert medical analyst specializing in lung disease evaluation."
        else:
            role_str = "You are an expert analyst specializing in the evaluation of the given samples."

        prompt_parts.append(f"""
    # Task: Factor Identification

    {role_str} Based on the contrasting pairs and interaction analysis:

    1. Identify the key factors that differentiate these samples
    2. Focus on factors that can be clearly observed. Each factor should focus on one concrete aspect without overlap{used_factors_hint}
    3. Create factors that have clear positive, neutral, and negative criteria

    # Output Format

    **Part 1**: Explain your observations about the key differences between each sample pair.

    **Part 2**: List the discovered factors using this exact template:
    '''
    **Factor Name**
    - 1: [Positive Criterion]
    - 0: [Otherwise; or not mentioned]
    - -1: [Negative Criterion]
    '''

    Aim to identify several distinct factors that show clear contrasts between samples.
    """)
        
        combined_image_path = image_path
        if mode in ["image-image", "text-image", "image-text"]:
            if self.model is None or self.model == "gemini":
                pass
            else:
                if mode == "image-image":
                    combined_image_path = [self.create_combined_batch_image(image_batch, n_cols=2)]
                else:
                    combined_image_path = [self.create_combined_batch_image(image_batch, n_cols=5)]

        return "".join(prompt_parts), combined_image_path

    def extract_factors_from_response(self, response):
        """
        Extract structured factors from LLM response with enhanced pattern matching
        """
        factors = []
        factor_names = []
        
        text_to_analyze = response
        
        lines = text_to_analyze.split('\n')
        current_factor = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            if line.strip().startswith('-'):
                bold_match = False
            else:
                bold_match = re.search(r'\*\*(.*?)\*\*', line)
            if bold_match:
                factor_name = bold_match.group(1).strip()
                
                if current_factor and current_factor['criteria'] and current_factor['name'] not in factor_names:
                    factors.append(current_factor)
                    factor_names.append(current_factor['name'])
                
                current_factor = {'name': factor_name, 'criteria': {}}
                continue
            
            if current_factor is None and re.match(r'^\d+\.\s+', line):
                has_criteria = False
                for j in range(1, 4):
                    if i+j < len(lines) and re.match(r'^-\s*[10-]:', lines[i+j].strip()):
                        has_criteria = True
                        break
                
                if has_criteria:
                    factor_name = re.sub(r'^\d+\.\s+', '', line).strip()
                    if '**' in factor_name:
                        bold_match = re.search(r'\*\*(.*?)\*\*', factor_name)
                        if bold_match:
                            factor_name = bold_match.group(1).strip()
                    
                    current_factor = {'name': factor_name, 'criteria': {}}
                    continue
            
            if current_factor:
                if re.match(r'^-\s*1:', line):
                    current_factor['criteria']['positive'] = re.sub(r'^-\s*1:', '', line).strip()
                elif re.match(r'^-\s*0:', line):
                    current_factor['criteria']['neutral'] = re.sub(r'^-\s*0:', '', line).strip()
                elif re.match(r'^-\s*-1:', line):
                    current_factor['criteria']['negative'] = re.sub(r'^-\s*-1:', '', line).strip()
                elif re.search(r'^-\s*\*\*1', line):
                    current_factor['criteria']['positive'] = re.sub(r'^-\s*\*\*1', '', line).strip()
                elif re.search(r'^-\s*\*\*0', line):
                    current_factor['criteria']['neutral'] = re.sub(r'^-\s*\*\*0', '', line).strip()
                elif re.search(r'^-\s*\*\*-1', line):
                    current_factor['criteria']['negative'] = re.sub(r'^-\s*\*\*-1', '', line).strip()
        
        if current_factor and current_factor['criteria'] and current_factor['name'] not in factor_names:
            factors.append(current_factor)
            factor_names.append(current_factor['name'])
        
        return factors

    def annotate_all_samples(self, meta, factors):
        """
        Annotate all samples using the identified factors
        """
        batch_size = 20
        batches = [meta.iloc[i:i+batch_size] for i in range(0, len(meta), batch_size)]
        
        all_annotations = {}

        for batch_idx, batch in enumerate(tqdm(batches, desc="Annotating batches", total=len(batches))):
            combined_image_path = self.create_combined_batch_image(batch, batch_idx)
            is_success = False
            for try_idx in range(10):
                try:
                    prompt = self.generate_annotation_prompt(batch, factors)
                    response = self.generate(prompt, img_path=[combined_image_path])
                    
                    batch_annotations = self.parse_annotation_response(response['text'])
                    
                    all_annotations.update(batch_annotations)
                    is_success = True
                    break
                except Exception as e:
                    print(f"Error generating annotations for batch {batch_idx}: {e}")
                    all_annotations[batch_idx] = {"error": str(e)}
            if not is_success:
                assert False, f"Failed to generate annotations for batch {batch_idx} after multiple attempts."
        
        return all_annotations
    
    def create_combined_batch_image(self, batch, batch_idx=None, n_cols=5):
        """
        Create a combined image with sample IDs for all samples in batch
        """
        images = []
        img_size = 256

        if isinstance(batch, list):
            batch = pd.DataFrame(batch)
        
        for i, (idx, row) in enumerate(batch.iterrows()):
            try:
                img = Image.open(row['ImagePath']).convert('RGB')
                
                draw = ImageDraw.Draw(img)
                
                font = None
                try:
                    font_paths = [
                        "arial.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                        "/Windows/Fonts/Arial.ttf"
                    ]
                    
                    for path in font_paths:
                        try:
                            font = ImageFont.truetype(path, 30)
                            break
                        except:
                            continue
                except:
                    pass
                    
                if font is None:
                    font = ImageFont.load_default()
                    
                text = f"ID: {idx}"
                text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (100, 30)
                padding = 5
                
                draw.rectangle(
                    [(5, 5), (text_width + 15, text_height + 15)],
                    fill=(0, 0, 0, 180)
                )
                
                draw.text((10, 10), text, fill=(255, 255, 255), font=font)
                
                images.append(img)
            except Exception as e:
                print(f"Error processing image {row['ImagePath']}: {e}")
                img = Image.new('RGB', (img_size, img_size), color=(255, 200, 200))
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 30)
                except:
                    font = ImageFont.load_default()
                    
                draw.rectangle([(5, 5), (150, 40)], fill=(0, 0, 0))
                draw.text((10, 10), f"ID: {idx}", fill=(255, 255, 255), font=font)
                
                draw.rectangle([(5, 45), (250, 80)], fill=(0, 0, 0))
                draw.text((10, 50), "Error loading image", fill=(255, 255, 255), font=font)
                
                images.append(img)
        
        num_images = len(images)
        cols = min(n_cols, num_images)
        rows = (num_images + cols - 1) // cols
        
        combined_img = Image.new('RGB', (img_size * cols, img_size * rows))
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            combined_img.paste(img.resize((img_size, img_size)), (col * img_size, row * img_size))
        
        if batch_idx is None:
            combined_path = os.path.join(self.output_path, f"combined_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        else:
            combined_path = os.path.join(self.output_path, f"batch_{batch_idx}_combined_image.jpg")
        combined_img.save(combined_path)
        
        return combined_path

    def generate_annotation_prompt(self, batch, factors):
        """
        Generate a prompt for annotating a batch of samples
        """
        prompt_parts = ["# Factor Annotation\n\n"]
        prompt_parts.append("I've provided a combined image showing all samples labeled with their sample IDs (ID: X in the top-left corner of each image).\n\n")
        prompt_parts.append("Please annotate the following samples based on these factors:\n\n")
        
        for idx, factor in enumerate(factors):
            prompt_parts.append(f"## Factor {idx+1}: {factor['name']}\n")
            prompt_parts.append(f"- 1: {factor['criteria']['positive']}\n")
            prompt_parts.append(f"- 0: {factor['criteria']['neutral']}\n")
            prompt_parts.append(f"- -1: {factor['criteria']['negative']}\n\n")
        
        prompt_parts.append("# Samples to Annotate\n\n")
        
        for i, (idx, row) in enumerate(batch.iterrows()):
            prompt_parts.append(f"## Sample {idx}\n")
            prompt_parts.append(f"- **Score**: {row['score']}\n")
            prompt_parts.append(f"- **Review**: {row['Review']}\n")
            prompt_parts.append(f"- **Image**: Look for the image labeled with ID: {idx} in the combined image\n\n")
        
        prompt_parts.append("""
    # Task
    For each sample, assign a value (-1, 0, or 1) to each factor based on the criteria above.

    # Output Format
    Please format your response strictly as follows, and '{factor1_name}' should be exactly the same as the factor name without any other characters:

    **Sample [ID]**:
    - Factor 1 ({factor1_name}): [Value]
    - Factor 2 ({factor2_name}): [Value]
    ...

    Repeat for all samples.
    """)
        
        return "".join(prompt_parts)

    def parse_annotation_response(self, response):
        """
        Parse the LLM's annotation response
        """
        annotations = {}
        lines = response.split('\n')
        current_sample = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('**Sample') and line.endswith('**:'):
                sample_id = int(line.replace('**Sample', '').replace('**:', '').strip())
                current_sample = sample_id
                annotations[current_sample] = {}
            
            elif current_sample is not None and line.startswith('- Factor'):
                parts = line.split(':')
                if len(parts) >= 2:
                    factor_part = parts[0].strip()
                    value_part = parts[-1].strip()
                    
                    factor_idx = int(factor_part.replace('- Factor', '').split('(')[0].strip()) - 1
                    
                    try:
                        value = int(value_part)
                        annotations[current_sample][factor_idx] = value
                    except:
                        if '1' in value_part:
                            annotations[current_sample][factor_idx] = 1
                        elif '-1' in value_part:
                            annotations[current_sample][factor_idx] = -1
                        else:
                            annotations[current_sample][factor_idx] = 0
        
        return annotations

    def find_mismatched_text_image_pairs(self, meta, text_embeddings, image_embeddings, alpha=1.0, beta=1.0, top_n=10):
        """
        Find mismatched text-image pairs using both CLIP embeddings and score differences
        
        Parameters:
        - alpha: Weight for CLIP embedding distance
        - beta: Weight for score difference
        """
        scores = meta['score'].values.astype(float)
        
        similarity = cosine_similarity(text_embeddings, image_embeddings)
        
        clip_distances = 1 - similarity
        
        N_sample = len(meta)
        score_diff_matrix = np.zeros((N_sample, N_sample))
        for i in range(N_sample):
            for j in range(N_sample):
                score_diff_matrix[i, j] = abs(scores[i] - scores[j])
        
        clip_distances = clip_distances / clip_distances.max()
        score_diff_matrix = score_diff_matrix / score_diff_matrix.max()
        
        combined_metric = alpha * clip_distances + beta * score_diff_matrix

        np.fill_diagonal(combined_metric, -np.inf)
        combined_flattened = combined_metric.flatten()
        top_indices = np.argsort(combined_flattened)[-top_n:]
        pairs = []
        for i in range(top_n):
            idx = top_indices[i]
            row = idx // combined_metric.shape[1]
            col = idx % combined_metric.shape[1]
            pairs.append((row, col, combined_flattened[idx]))
        
        return pairs

    def generate_mismatch_prompt(self, meta, pairs, used_factors=None):
        """
        Generate a prompt focusing on mismatches between text and image
        """
        prompt_parts = []
        image_paths = []

        used_factors_hint = ""
        if used_factors is not None and used_factors:
            used_factors_hint = "\n - Avoid overlapping with **those existing factors:**\n\t- " + '\n\t- '.join(used_factors) + '\n'
        
        prompt_parts.append("# Inter-modal Mismatch Analysis\n\n")
        prompt_parts.append("I'm showing you samples where the textual description (review) and visual appearance (image) of samples seem to contradict each other.\n\n")
        
        for idx, (text_idx, image_idx, mismatch_score) in enumerate(pairs[:5]):  # Limit to 5 pairs
            prompt_parts.append(f"\n## Pair {idx+1} (Mismatch Score: {mismatch_score:.4f})\n")
            
            prompt_parts.append(f"### Text from Sample {text_idx}\n")
            prompt_parts.append(f"- **Score**: {meta.iloc[text_idx]['score']}\n")
            prompt_parts.append(f"- **Review**: {meta.iloc[text_idx]['Review']}\n")
            
            prompt_parts.append(f"\n### Image from Sample {image_idx}\n")
            prompt_parts.append(f"- **Score**: {meta.iloc[image_idx]['score']}\n")
            
            image_path = meta.iloc[image_idx]['ImagePath']
            image_paths.append(image_path)
            prompt_parts.append(f"- **Image**: [Image {len(image_paths)}] (The image is provided with this prompt)\n")
            
            prompt_parts.append("\n**Notice the potential mismatch between what the review describes and what the image shows. Analyze the underlying interactions among potential factors that could lead to this mismatch.**\n")

        if self.dataset == 'MAG9':
            role_str = "You are an expert food analyst specializing in apple evaluation."
        elif self.dataset == 'Lung4':
            role_str = "You are an expert medical analyst specializing in lung disease evaluation."
        else:
            role_str = "You are an expert analyst specializing in the evaluation of the given samples."
        
        prompt_parts.append(f"""
    # Task: Identify Factors that Explain Text-Image Discrepancies

    {role_str} Based on the constrastive pairs and interaction analysis:

    1. Identify key factors where the textual reviews contradict what's visible in the paired images
    2. Each factor should focus on one concrete aspect without overlap{used_factors_hint}
    3. Create factors that can explain these discrepancies with clear positive, neutral, and negative criteria

    # Output Format

    **Part 1**: Explain your observations about the mismatches of each pair of textual and visual information.

    **Part 2**: List factors that explain these discrepancies using this exact template:
    '''
    **Factor Name**
    - 1: [Positive Criterion]
    - 0: [Otherwise; or not mentioned]
    - -1: [Negative Criterion]
    '''

    Aim to identify a few distinct factors that could explain text-image mismatches in evaluations.
    """)
        
        return "".join(prompt_parts), image_paths

    def deduplicate_factors_with_llm(self, factors, iter=1, last_factors=None):
        """
        Use LLM to deduplicate and refine the factors
        """
        factor_text = ""
        
        if iter > 1 and last_factors is not None:
            factor_text += f"Note that, here are the previously identified factors:\n"
            for i, factor in enumerate(last_factors):
                factor_text += f"Factor {i+1}: {factor['name']}\n"
                factor_text += f"- 1: {factor['criteria'].get('positive', 'N/A')}\n"
                factor_text += f"- 0: {factor['criteria'].get('neutral', 'N/A')}\n"
                factor_text += f"- -1: {factor['criteria'].get('negative', 'N/A')}\n\n"
            factor_text += f"Together with the following identified factors in this iteration, please refine and deduplicate the factors. They should not have any explicit and semantic overlap with previously defined factors and with each other. You need to reduce the number of factors and output the deduplicated factors from the previous and the following ones.\n\n"

        for i, factor in enumerate(factors):
            factor_text += f"Factor {i+1}: {factor['name']}\n"
            factor_text += f"- 1: {factor['criteria'].get('positive', 'N/A')}\n"
            factor_text += f"- 0: {factor['criteria'].get('neutral', 'N/A')}\n"
            factor_text += f"- -1: {factor['criteria'].get('negative', 'N/A')}\n\n"

        

        
        prompt = f"""
    # Factor Deduplication and Refinement

    Below is a list of factors identified from different contrastive analyses:

    {factor_text}

    Please:
    1. Identify and merge similar factors; Each factor should be distinct and not overlap with others and be specific to one aspect; Avoid general factors like "overall quality"
    2. Refine factor definitions for clarity and precision
    3. Output a consolidated list of the most important and non-overlapping factors

    Only output the final results using this exact format for each factor:
    '''
    **Factor Name** (Names only)
    - 1: [Positive Criterion]
    - 0: [Otherwise; or not mentioned]
    - -1: [Negative Criterion]
    '''
    """
        
        response = self.generate(prompt)
        consolidated_factors = self.extract_factors_from_response(response['text'])
        
        return consolidated_factors

    def analyze_factors_from_contrastive_pairs(self, meta, alpha=1.0, beta=1.0, used_factors=None, iter=1):
        """
        Extract factors using contrastive analysis with CLIP embeddings
        """
        text_embeddings, image_embeddings = self.extract_clip_embeddings(meta)
        
        text_text_pairs = self.compute_distance_pairs(text_embeddings)
        image_image_pairs = self.compute_distance_pairs(image_embeddings)
        
        mismatched_pairs = self.find_mismatched_text_image_pairs(
            meta, text_embeddings, image_embeddings, alpha, beta, top_n=self.top_n
        )
        
        if used_factors is None:
            all_factors = []
            last_factors = None
            new_factors = []
        else:
            all_factors = used_factors.copy()
            last_factors = used_factors.copy()
            new_factors = []

        if os.path.exists(os.path.join(self.output_path, f"iter{iter}_deduplicated_factors_response.json")):
            with open(os.path.join(self.output_path, f"iter{iter}_deduplicated_factors_response.json"), 'r') as f:
                response = json.load(f)
                unique_factors = self.extract_factors_from_response(response['text'])
        elif os.path.exists(os.path.join(self.output_path, "deduplicated_factors_response.json")):
            with open(os.path.join(self.output_path, "deduplicated_factors_response.json"), 'r') as f:
                response = json.load(f)
                unique_factors = self.extract_factors_from_response(response['text'])
        else:
            print("Analyzing text-text contrasts...")
            prompt, _ = self.generate_contrastive_prompt(meta, text_text_pairs, mode="text-text", used_factors=[f['name'] for f in all_factors])

            response = self.generate(prompt)
            text_text_factors = self.extract_factors_from_response(response['text'])
            assert len(text_text_factors) > 0, "Text-text factors should not be empty."
            all_factors.extend(text_text_factors)
            new_factors.extend(text_text_factors)
            print(f"Identified {len(text_text_factors)} text-text factors: {[f['name'] for f in text_text_factors]}")

            print("Analyzing image-image contrasts...")
            prompt, image_paths = self.generate_contrastive_prompt(meta, image_image_pairs, mode="image-image", used_factors=[f['name'] for f in all_factors])
            response = self.generate(prompt, img_path=image_paths)
            image_image_factors = self.extract_factors_from_response(response['text'])
            assert len(image_image_factors) > 0, "Image-image factors should not be empty."
            all_factors.extend(image_image_factors)
            new_factors.extend(image_image_factors)
            print(f"Identified {len(image_image_factors)} image-image factors: {[f['name'] for f in image_image_factors]}")
            
            print("Analyzing text-image mismatches...")
            prompt, image_paths = self.generate_mismatch_prompt(meta, mismatched_pairs, used_factors=[f['name'] for f in all_factors])
            response = self.generate(prompt, img_path=image_paths)
            mismatch_factors = self.extract_factors_from_response(response['text'])
            assert len(mismatch_factors) > 0, "Text-image mismatch factors should not be empty."
            all_factors.extend(mismatch_factors)
            new_factors.extend(mismatch_factors)
            print(f"Identified {len(mismatch_factors)} text-image mismatch factors: {[f['name'] for f in mismatch_factors]}")
            unique_factors = self.deduplicate_factors_with_llm(new_factors, iter=iter, last_factors=last_factors)
        assert len(unique_factors) > 0, "Unique factors should not be empty."
        pairs = {}
        pairs['text_text_pairs'] = text_text_pairs
        pairs['image_image_pairs'] = image_image_pairs
        pairs['mismatched_pairs'] = mismatched_pairs
        return unique_factors, pairs

    def run_contrastive_factor_analysis(self, meta, used_factors=None, alpha=1.0, beta=1.0, iter=1):
        
        print("Identifying factors through contrastive analysis...")

        factors, pairs = self.analyze_factors_from_contrastive_pairs(meta, alpha=alpha, beta=beta, used_factors=used_factors, iter=iter)
        
        print(f"Identified {len(factors)} unique new factors in iter{iter}.")
        if len(factors) > 0:
            for i, factor in enumerate(factors):
                print(f"{i+1}. {factor['name']}")
            
            print("Annotating all samples with identified factors...")
            if not os.path.exists(os.path.join(self.output_path, f"iter{iter}_annotation_results.csv")):
                annotations = self.annotate_all_samples(meta, factors)
            
                annotation_df = pd.DataFrame(0, index=range(self.N_sample), columns=[f['name'] for f in factors])
                
                for sample_id, factor_values in annotations.items():
                    for factor_idx, value in factor_values.items():
                        factor_name = factors[factor_idx]['name']
                        annotation_df.loc[sample_id, factor_name] = value
            else:
                annotation_df = pd.read_csv(os.path.join(self.output_path, f"iter{iter}_annotation_results.csv"), index_col=0)
                
        return factors, annotation_df, pairs

    def update(self, data_interface, data_interface2):
        for idx in data_interface.index:
            for col in data_interface2.columns:
                data_interface.loc[idx, col] = data_interface2.loc[idx, col]
        return data_interface

    def get_uncertain_factors(self, annotated_name, g):
        index_to_name = {f"X{i+1}": name for i, name in enumerate(annotated_name)}

        uncertain_edges = []
        uncertain_factors = set()
        nodes_in_uncertain_edges = set()

        for edge in g.get_graph_edges():
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            end1 = edge.get_endpoint1()
            end2 = edge.get_endpoint2()

            if Endpoint.CIRCLE in (end1, end2):
                node1_idx = node1.get_name()
                node2_idx = node2.get_name()
                edge_info = {
                    "from": index_to_name[node1_idx],
                    "to": index_to_name[node2_idx],
                    "endpoint1": end1.name,
                    "endpoint2": end2.name
                }
                uncertain_edges.append(edge_info)
                uncertain_factors.add(index_to_name[node1_idx])
                uncertain_factors.add(index_to_name[node2_idx])
                nodes_in_uncertain_edges.add(node1_idx)
                nodes_in_uncertain_edges.add(node2_idx)

        all_nodes_obj = g.get_nodes()
        all_node_indices = {node.get_name() for node in all_nodes_obj}

        adj = collections.defaultdict(set)
        nodes_with_edges = set()
        for edge in g.get_graph_edges():
            node1_idx = edge.get_node1().get_name()
            node2_idx = edge.get_node2().get_name()
            adj[node1_idx].add(node2_idx)
            adj[node2_idx].add(node1_idx)
            nodes_with_edges.add(node1_idx)
            nodes_with_edges.add(node2_idx)

        isolated_node_indices = all_node_indices - nodes_with_edges
        for node_idx in isolated_node_indices:
            if node_idx in index_to_name:
                uncertain_factors.add(index_to_name[node_idx])

        visited = set()
        components = []
        for node_idx in nodes_with_edges:
            if node_idx not in visited:
                component_indices = set()
                q = collections.deque([node_idx])
                visited.add(node_idx)
                while q:
                    curr = q.popleft()
                    component_indices.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                if component_indices:
                    components.append(component_indices)

        if len(components) > 1:
            components.sort(key=len, reverse=True)
            largest_component_indices = components[0]
            
            for comp_indices in components[1:]:
                for node_idx in comp_indices:
                    if node_idx in index_to_name:
                        uncertain_factors.add(index_to_name[node_idx])
        
        if 'score' in uncertain_factors:
            uncertain_factors.remove('score')

        return uncertain_factors
    
    
    def generate_causal_refinement_prompt(self, annotated_name, uncertain_factors, data_interface, meta, pairs, pdy=None):
        if pdy is not None:
            pdy_str = f"\nHere is the discovered causal graph for your reference: {pdy.to_string()}"
        else:
            pdy_str = "\n"
        
        factors_text = "\n".join([f"- {factor}" for factor in annotated_name])
        uncertain_factors_text = "\n".join([f"- {factor}" for factor in uncertain_factors])

        data_section = "\nHere are the selected samples for your reference:\n\n"
        image_paths = []
        selected_idx = set()

        for pair_item in pairs.items():
            pair_item = pair_item[1]
            for i in range(min(self.top_n, len(pair_item))):
                idx_a = pair_item[i][0]
                idx_b = pair_item[i][1]
                selected_idx.add(idx_a)
                selected_idx.add(idx_b)
        
        for idx in selected_idx:
            review = meta.iloc[idx]['Review']
            image_path = meta.iloc[idx]['ImagePath']
            image_paths.append(image_path)
            data_section += f"### Sample {idx}\n"
            data_section += f"- **Review**: {review}\n"
            data_section += f"- **Image**: [Sample Image {len(image_paths)}] (The image is provided with this prompt)\n\n"
        selected_data_interface = data_interface.iloc[list(selected_idx)]
        selected_data_interface = selected_data_interface[annotated_name]
        
        if self.dataset == 'MAG9':
            obj_str = "apples"
        elif self.dataset == 'Lung4':
            obj_str = "lung disease"
        else:
            obj_str = "analysis"

        prompt = f"""
# Refining Causal Factors and Relationships

I'm analyzing factors that affect sample scores based on reviews and images. I need your help to refine my understanding of causal relationships.

## Data
{data_section}

## Current Factors
{factors_text} {pdy_str}

## Annotated Factors
{selected_data_interface}

## Uncertain Causal Relationships
I've identified these causal factors, but there is uncertainty in the causal relationships of the following factors:
{uncertain_factors_text}

# Task: Counterfactual Reasoning

For each uncertain relationship above, please create a counterfactual scenario: "If factor X were different (i.e., value being reversed if the factor is mentioned and skip the sample if the factor is not mentioned for this sample), how would other factors be affected?" Based on this assumption and your knowledge of {obj_str}, predict the values of other factors. Only create counterfactual scenarios that support the valid and reasonable causal relationships and directions. Specifically, there are two types of factors, verbal and visual factors. If the verbal factor is modified, revise the review text directly with minimum changes to reflect the new scenario. If the visual factor is modified, state a short instruction of the changes that need to be applied to the image, e.g., "Change the color of the apple to red." If no visual changes, please state 'N/A'.

# Output Format
Please structure your response in the following format and respectively list the values of all the factors for each sample in each counterfactual scenario:
```
## Counterfactual Scenario 1: [Changed Factor Name]
**Sample 1**:
- Factor 1: [Value]
- Factor 2: [Value]
...
- Factor N: [Value]
- Review: [Modified Review Text]
- Image: [Image Modification Description] (If applicable, otherwise 'N/A')

**Sample 2**:
...

## Counterfactual Scenario 2: [Changed Factor Name]
...
```
        """
        
        return prompt, image_paths
    
    def generate_causal_refinement_response(self, counterfactual_prompt, counterfactual_img_paths, annotated_name, meta, iter=1, id_start=0):
        if not os.path.exists(os.path.join(self.output_path, f"iter{iter}_cf_pd.csv")) and not os.path.exists(os.path.join(self.output_path, "cf_pd.csv")):
            counterfactual_response = self.generate(prompt=counterfactual_prompt, img_path=counterfactual_img_paths)
        
            if self.dataset == 'Lung4':
                counterfactual_path = './lung_images_counterfactual'
            else:
                counterfactual_path = './apple_images_counterfactual'
            os.makedirs(counterfactual_path, exist_ok=True)

            from counterfactual_utils import extract_counterfactual_data

            columns = annotated_name + ['Review', 'Image']
            cf_scenarios, cf_pd_extracted = extract_counterfactual_data(counterfactual_response["text"], columns)

            cf_pd = pd.DataFrame(cf_pd_extracted)
            last_factor = ''

            for i in range(len(cf_pd)):
                if last_factor != cf_pd.loc[i, 'scenario']:
                    id_start += 1000
                    last_factor = cf_pd.loc[i, 'scenario']
                sample_id = int(cf_pd.loc[i, 'sample_id'])
                if 'N/A' in str(cf_pd.loc[i, 'Image']) or str(cf_pd.loc[i, 'Image']) == '':
                    cf_pd.loc[i, 'Image'] = meta.loc[sample_id, 'ImagePath']
                else:
                    cf_sample_image_prompt = cf_pd.loc[i, 'Image']
                    sample_ori_image = meta.loc[sample_id, 'ImagePath']
                    cf_image_response = self.img_generate(prompt=cf_sample_image_prompt, img_path=[sample_ori_image], img_save_dir=counterfactual_path)
                    if cf_image_response['images'] == []:
                        cf_pd.loc[i, 'Image'] = sample_ori_image
                    else:
                        cf_pd.loc[i, 'Image'] = cf_image_response['images'][0]
                cf_pd.loc[i, 'sample_id'] = id_start + sample_id
                
        else:
            if os.path.exists(os.path.join(self.output_path, f"iter{iter}_cf_pd.csv")):
                cf_pd = pd.read_csv(os.path.join(self.output_path, f"iter{iter}_cf_pd.csv"))
            else:
                cf_pd = pd.read_csv(os.path.join(self.output_path, "cf_pd.csv"))
        
        return cf_pd

    def compute_text_similarity(self, original_text, cf_text):
        with torch.no_grad():
            orig_tokens = clip.tokenize([original_text]).to(device)
            cf_tokens = clip.tokenize([cf_text[:min(len(cf_text), 77)]]).to(device)
            with torch.no_grad():
                orig_emb = model.encode_text(orig_tokens)
                cf_emb = model.encode_text(cf_tokens)
        return cosine_similarity(orig_emb.cpu(), cf_emb.cpu())[0][0]

    def compute_image_similarity(self, orig_path, cf_path):
        orig_img = preprocess(Image.open(orig_path)).unsqueeze(0).to(device)
        cf_img = preprocess(Image.open(cf_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_emb = model.encode_image(orig_img)
            cf_emb = model.encode_image(cf_img)
        return cosine_similarity(orig_emb.cpu(), cf_emb.cpu())[0][0]

    def check_counterfactual_generation(self, cf_pd, meta, data_interface, annotated_name, g, clip_threshold=0.7, causal_threshold_consistency=0.4):
        validated_cf_pd_rows_causal = []
        rejected_count_causal = 0
        validation_results_causal = []

        original_data_for_corr = data_interface[annotated_name]

        for index, cf_row in tqdm(cf_pd.iterrows(), total=len(cf_pd), desc="Validating CF Samples (Causal)"):
            cf_sample_id = cf_row['sample_id']
            original_sample_idx = int(cf_sample_id) % 1000

            if original_sample_idx not in data_interface.index:
                rejected_count_causal += 1
                validation_results_causal.append({
                    'sample_id': cf_sample_id,
                    'status': 'rejected_original_missing',
                    'details': {}
                })
                continue

            original_row = data_interface.loc[original_sample_idx]
            original_sample = meta.iloc[original_sample_idx]
            
            review = original_sample['Review']
            text_sim = self.compute_text_similarity(review[:min(len(review), 77)], cf_row['Review'])
            img_sim = self.compute_image_similarity(original_sample['ImagePath'], cf_row['Image'])
            similarity_score = (text_sim + img_sim) / 2

            is_clip_valid = similarity_score >= clip_threshold
            clip_details = {
                'sample_id': cf_row['sample_id'],
                'text_sim': text_sim,
                'img_sim': img_sim,
                'is_clip_valid': is_clip_valid
            }
            if not is_clip_valid:
                clip_details['reason'] = "CLIP similarity below threshold"

            is_consistent, details = validate_causal_consistency(
                cf_row=cf_row,
                original_row=original_row,
                graph=g,
                labels=annotated_name,
                threshold_independent_change=causal_threshold_consistency,
            )

            if is_consistent and is_clip_valid:
                validated_cf_pd_rows_causal.append(cf_row.to_dict())
                validation_results_causal.append({
                    'sample_id': cf_sample_id,
                    'status': 'accepted_clip_causal',
                    'clip_details': clip_details,
                    'causal_details': details
                })
            else:
                if not is_clip_valid:
                    validation_results_causal.append({
                        'sample_id': cf_sample_id,
                        'status': 'rejected_clip',
                        'clip_details': clip_details
                    })
                if not is_consistent:
                    validation_results_causal.append({
                        'sample_id': cf_sample_id,
                        'status': 'rejected_causal',
                        'details': details
                    })
                rejected_count_causal += 1

        validated_cf_pd_causal = pd.DataFrame(validated_cf_pd_rows_causal)

        return validated_cf_pd_causal

