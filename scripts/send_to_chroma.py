#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import chromadb
import argparse
import pandas as pd
from datetime import datetime

def load_reduced_data(input_file: str) -> tuple[np.ndarray, list[str]]:
    """
    Load the reduced embeddings and labels from the CSV file.
    
    Args:
        input_file: Path to the CSV file containing reduced embeddings
        
    Returns:
        tuple: (reduced_embeddings, labels) where reduced_embeddings is a numpy array
               and labels is a list of strings
    """
    try:
        df = pd.read_csv(input_file)
        numeric_cols = df.select_dtypes(include='number').columns
        reduced_embeddings = df[numeric_cols].values
        
        labels = df['label'].tolist() if 'label' in df.columns else []
        
        return reduced_embeddings, labels
    except Exception as e:
        raise ValueError(f"Error loading reduced data: {str(e)}")

def send_to_chroma(input_file: str, collection_name: str, method: str, params: dict, chroma_host: str, chroma_port: str) -> None:
    """
    Send reduced embeddings to ChromaDB.
    
    Args:
        input_file: Path to the CSV file containing reduced embeddings
        collection_name: Name of the ChromaDB collection to create
        method: Dimensionality reduction method used
        params: Dictionary of parameters used for reduction
        chroma_host: ChromaDB host address
        chroma_port: ChromaDB port number
    """
    try:
        try:
            client = chromadb.HttpClient(
                host=chroma_host,
                port=int(chroma_port)
            )
            client.heartbeat()
        except Exception as e:
            print(f"Error connecting to ChromaDB: {str(e)}", file=sys.stderr)
            sys.exit(1)
        
        reduced_embeddings, labels = load_reduced_data(input_file)

        print("reduced_embeddings", reduced_embeddings)
        print("labels", labels)
        
        metadata = {
            "type": "saved",
            "method": method,
            "params": json.dumps(params),
            "created": str(datetime.now())
        }
        
        client.create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=None,
            get_or_create=True
        )
        
        metadatas = [{"label": label} for label in labels]
        
        collection = client.get_collection(collection_name)

        collection.add(
            embeddings=reduced_embeddings.tolist(),
            ids=[f"reduced_{i}" for i in range(len(reduced_embeddings))],
            metadatas=metadatas
        )
        
        print(f"Successfully added reduced embeddings to ChromaDB collection: {collection_name}")
        
    except Exception as e:
        print(f"Error sending data to ChromaDB: {str(e)}", file=sys.stderr)
        sys.exit(1)

def create_temp_script(input_file: str, collection_name: str, method: str, params: dict, chroma_host: str, chroma_port: str) -> str:
    """
    Create a temporary script with environment variables and command.
    
    Args:
        input_file: Path to the input file
        collection_name: Name of the ChromaDB collection
        method: Dimensionality reduction method
        params: Dictionary of parameters
        chroma_host: ChromaDB host address
        chroma_port: ChromaDB port number
        
    Returns:
        str: Path to the temporary script
    """
    script_content = f"""#!/bin/bash
export CHROMA_HOST="{chroma_host}"
export CHROMA_PORT="{chroma_port}"

python3 {os.path.join(os.path.dirname(__file__), 'send_to_chroma.py')} \\
    "{input_file}" \\
    "{collection_name}" \\
    "{method}" \\
    '{json.dumps(params)}' \\
    "$CHROMA_HOST" \\
    "$CHROMA_PORT"
"""
    
    temp_script = os.path.join(os.path.dirname(input_file), "temp_send_to_chroma.sh")
    with open(temp_script, 'w') as f:
        f.write(script_content)
    os.chmod(temp_script, 0o755)
    
    return temp_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send reduced embeddings to ChromaDB")
    parser.add_argument("input_file", help="Path to the CSV file containing reduced embeddings")
    parser.add_argument("collection_name", help="Name of the ChromaDB collection to create")
    parser.add_argument("method", help="Dimensionality reduction method used")
    parser.add_argument("params", help="JSON string of parameters used for reduction")
    parser.add_argument("chroma_host", help="ChromaDB host address")
    parser.add_argument("chroma_port", help="ChromaDB port number")
    
    args = parser.parse_args()
    
    try:
        params = json.loads(args.params)
        send_to_chroma(
            args.input_file,
            args.collection_name,
            args.method,
            params,
            args.chroma_host,
            args.chroma_port
        )
    except json.JSONDecodeError:
        print("Error: params must be a valid JSON string", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
