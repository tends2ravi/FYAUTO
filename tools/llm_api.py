"""
LLM API tool for interacting with various providers.
"""
import argparse
import json
import os
import sys
from typing import Optional
import google.generativeai as genai

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interact with various LLM providers"
    )
    
    parser.add_argument(
        "--prompt",
        required=True,
        help="The prompt to send to the LLM"
    )
    
    parser.add_argument(
        "--provider",
        choices=["gemini"],
        default="gemini",
        help="The LLM provider to use"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the response to"
    )
    
    return parser.parse_args()

def call_gemini(prompt: str) -> str:
    """Call the Gemini API."""
    try:
        # Configure the Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
            
        genai.configure(api_key=api_key)
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                candidate_count=1,
                max_output_tokens=2000,
            )
        )
        
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}", file=sys.stderr)
        raise

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.provider == "gemini":
            response = call_gemini(args.prompt)
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")
        
        if args.output:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response)
        else:
            print(response)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    main() 