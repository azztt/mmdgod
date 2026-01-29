#!/usr/bin/env python3
"""
Sample a subset of images from a COCO-style manifest for faster testing.
Keeps all annotations for the sampled images and preserves categories.
"""
import json
import argparse
import random
from pathlib import Path


def sample_manifest(input_path, output_path, num_images=500, seed=42):
    """Sample a subset of images and their annotations."""
    print(f"Loading manifest from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    print(f"Original: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    
    # Sample images
    random.seed(seed)
    num_to_sample = min(num_images, len(images))
    sampled_images = random.sample(images, num_to_sample)
    
    # Get IDs of sampled images
    sampled_image_ids = set(img['id'] for img in sampled_images)
    
    # Filter annotations to only include those for sampled images
    sampled_annotations = [
        ann for ann in annotations 
        if ann.get('image_id') in sampled_image_ids
    ]
    
    # Create output manifest
    output_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': categories  # Keep all categories
    }
    
    # Copy any other top-level keys
    for key in data:
        if key not in ['images', 'annotations', 'categories']:
            output_data[key] = data[key]
    
    print(f"Sampled: {len(sampled_images)} images, {len(sampled_annotations)} annotations")
    
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")
    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a manifest for faster testing')
    parser.add_argument('input', help='Input manifest path')
    parser.add_argument('--output', '-o', help='Output manifest path (default: input_sampled.json)')
    parser.add_argument('--num-images', '-n', type=int, default=500, help='Number of images to sample')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_sampled{input_path.suffix}"
    
    sample_manifest(input_path, output_path, args.num_images, args.seed)
