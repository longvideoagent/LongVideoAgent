#evaluate_api_tvqa_plus.py
import argparse
from utils import run_simple_qa
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TVQA evaluation tool using Gemini model"
    )
    
    parser.add_argument(
        "--questions-path", 
        required=True,
        help="Path to questions JSON file (required)"
    )
    
    parser.add_argument(
        "--subs-path", 
        required=True,
        help="Path to subtitles JSON file (required)"
    )
    
    parser.add_argument(
        "--output-filename", 
        required=True,
        help="Output filename for results (required)"
    )
    
    parser.add_argument(
        "--base-frame-dir", 
        required=True,
        help="Directory containing frame images (required)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Model name to use (required)"
    )
    
    parser.add_argument(
        "--threads", 
        type=int,
        required=True,
        help="Number of threads to use (required)"
    )
    
    return parser.parse_args()

def main():
    """Main function to run TVQA evaluation"""
    args = parse_arguments()
    
    # Validate required paths exist
    required_paths = {
        "Questions path": args.questions_path,
        "Subtitles path": args.subs_path,
        "Base frame directory": args.base_frame_dir
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            print(f"Error: {name} does not exist: {path}")
            return
    
    print("Starting TVQA evaluation...")
    print(f"Questions path: {args.questions_path}")
    print(f"Subtitles path: {args.subs_path}")
    print(f"Base frame directory: {args.base_frame_dir}")
    print(f"Output file: {args.output_filename}")
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    
    run_simple_qa(
        questions_path=args.questions_path,
        subs_path=args.subs_path,
        output_filename=args.output_filename,
        base_frame_dir=args.base_frame_dir,
        model=args.model,
        num_threads=args.threads
    )

if __name__ == "__main__":
    main()