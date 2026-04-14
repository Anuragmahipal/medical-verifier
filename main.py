import argparse
import os
import requests

from atomic import extract_atomic_claims
from spo_graph import process_reasoning_to_graph
from graph_approach import visualize_graph



def check_ollama():
    try:
        requests.get("http://localhost:11434")
        return True
    except:
        return False
    



def read_input_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return content


def main():

    """ 
    We are using the parser to get arguements in the terminal, like:
        main.py --input "random_file_name_idk.txt"
        this will read the input from that file

        Arguments are:
            --input_file : specify the input file that we need
            --visualize : boolean flag to specify if visualization is needed

    NOTE : Please add proper documentation if more arguments are added , it really helps :)

    """

    parser = argparse.ArgumentParser(description="Medical Reasoning Graph Generator")

    parser.add_argument("--input_file", type=str, help="Path to input text file")
    parser.add_argument("--visualize", action="store_true", help="Visualize graph")

    args = parser.parse_args()

    
    if not check_ollama():
        print("Ollama is not running, please ensure that ollama is running")
        return 

    if args.input_file:
        reasoning = read_input_from_file(args.input_file)
        print(f"[INFO] Loaded input from {args.input_file}")
    else:
        # IF no input file was provided, we fallback to terminal input
        print("Enter reasoning (end with empty line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        reasoning = "\n".join(lines)

    
    print("\n[STEP 1] Extracting atomic claims...")
    claims = extract_atomic_claims(reasoning)
    print(f"[INFO] Extracted {len(claims)} claims")

    # -------- Step 2: Build Graph --------
    print("\n[STEP 2] Building reasoning graph...")
    graph_data, output_path = process_reasoning_to_graph(reasoning)

    print(f"[INFO] Graph saved to: {output_path}")

    # -------- Step 3: Visualization --------
    if args.visualize:
        print("\n[STEP 3] Visualizing graph...")
        image_path = output_path.replace(".json", ".png")
        visualize_graph(output_path, image_path)
        print(f"[INFO] Graph image saved to: {image_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()