from typing import Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import umap
import trimap
import pacmap
from chromadb import GetResult
from chromadb.api.models.Collection import Collection
from sklearn.manifold import TSNE
import numpy as np
import json
import tempfile
import os

from large_scale_dimensionality_reduction.vector_db import VectorDB
from large_scale_dimensionality_reduction.embeddings import Embeddings
from large_scale_dimensionality_reduction.utils import S3Client, DatasetDB, SSHClient, setup_logger

logger = setup_logger("dim_reduction-logger")

def create_embeddings(embeddings_instance: Embeddings, uploaded_file, label_column: str = "label", description: str = None) -> str | None:
    """
    Creates embeddings from uploaded CSV file and adds them to the vector database.
    Also uploads the file to S3 for backup and stores metadata in SQLite.

    Parameters:
    embeddings_instance : Embeddings
        The embeddings instance to use for generating embeddings.
    uploaded_file : file object
        CSV file uploaded by the user. Must contain:
        - a 'text' column with the text data
        - a column specified by label_column containing the labels (if not named 'label')
        - optionally an 'id' column for custom document IDs (if not present, will be auto-generated)
    label_column : str
        Name of the column containing the labels. Defaults to "label". If provided, this name will be used
        as the key in the metadata dictionary.
    description : str, optional
        Optional description of the dataset.

    Returns:
    str or None
        The name of the created collection, or None if there was an error.
    """
    try:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
            return None

        if label_column not in df.columns:
            st.error(f"The CSV file must contain a '{label_column}' column.")
            return None
        
        collection_name = uploaded_file.name[:-4]
        
        labels = df[label_column].fillna("unknown").astype(str).tolist()
        texts = df["text"].tolist()
        ids = [f"doc_{i}" for i in range(len(texts))] if "id" not in df.columns else df["id"].tolist()
        metadatas = [{label_column: label} for label in labels]

        s3_key = None
        try:
            s3_client = S3Client()
            s3_key = s3_client.upload_dataframe(
                df=df,
                filename=uploaded_file.name,
                prefix="raw_data"
            )
            st.success(f"Dataset uploaded to S3: {s3_key}")
        except Exception as e:
            st.error(f"Failed to upload dataset to S3: {str(e)}")
            return None

        if s3_key:
            try:
                db = DatasetDB()
                db.add_dataset(
                    name=uploaded_file.name,
                    s3_key=s3_key,
                    collection_name=collection_name,
                    label_column=label_column,
                    num_rows=len(df),
                    description=description
                )
                st.success("Dataset information stored in local database")
            except Exception as e:
                st.error(f"Failed to store dataset information in local database: {str(e)}")
                return None

        try:
            embeddings_instance.batch_process_texts(
                texts=texts,
                collection_name=collection_name,
                metadatas=metadatas,
                ids=ids,
                batch_size=5000
            )
            st.success(f"Embeddings stored in ChromaDB collection: {collection_name}")
        except Exception as e:
            st.error(f"Failed to store embeddings in ChromaDB: {str(e)}")
            if s3_key:
                try:
                    db = DatasetDB()
                    db.delete_dataset(db.get_dataset_by_s3_key(s3_key)["id"])
                    
                    s3_client = S3Client()
                    s3_client.delete_object(s3_key)
                    st.info(f"Cleaned up: removed dataset from S3 and local database")
                except Exception as cleanup_error:
                    st.warning(f"Failed to clean up resources: {str(cleanup_error)}")
            return None

        return collection_name, s3_key

    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return None


def download_dataset_from_s3(s3_key: str) -> pd.DataFrame | None:
    """
    Download a dataset from S3.
    
    Args:
        s3_key: The S3 key of the dataset to download
        
    Returns:
        pd.DataFrame | None: The downloaded dataset or None if download failed
    """
    try:
        s3_client = S3Client()
        df = s3_client.download_dataframe(s3_key)
        return df
    except Exception as e:
        st.error(f"Failed to download dataset from S3: {str(e)}")
        return None


def get_embeddings(db: VectorDB, dataset_name: str) -> Tuple[np.ndarray, list[str]]:
    """
    Returns embeddings and corresponding labels from the database.

    Parameters:
    db : VectorDB
        The vector database instance for retrieving embeddings.
    dataset_name : str
        Name of the collection to retrieve embeddings from.

    Returns:
    tuple
        (embeddings, labels): The retrieved embeddings as numpy array and corresponding labels.
    """

    db_collection = db.get_all_items_from_collection(dataset_name, include=["embeddings", "metadatas"])

    embeddings = np.array(db_collection["embeddings"])

    metadatas = db_collection["metadatas"]

    if metadatas and len(metadatas) > 0:
        label_key = next(iter(metadatas[0].keys()))
        labels = [metadata.get(label_key) for metadata in metadatas]
    else:
        labels = []

    return embeddings, labels


def transfer_script_to_hpc(method: str, dataset_filename: str = None) -> bool:
    """
    Transfer the appropriate visualization script, its slurm job script, and download script to HPC server.
    
    Args:
        method: The dimensionality reduction method ('UMAP', 't-SNE', 'PaCMAP', or 'TriMAP')
        dataset_filename: Optional name of the dataset file in S3 to download
        
    Returns:
        bool: True if transfer was successful, False otherwise
    """
    script_map = {
        'UMAP': ('visualisations_script_umap.py', 'slurm_job_umap.sh'),
        't-SNE': ('visualisations_script_tsne.py', 'slurm_job_tsne.sh'),
        'PaCMAP': ('visualisations_script_pacmap.py', 'slurm_job_pacmap.sh'),
        'TriMAP': ('visualisations_script_trimap.py', 'slurm_job_trimap.sh')
    }
    
    if method not in script_map:
        logger.error(f"Unsupported method: {method}")
        return False
        
    python_script, slurm_script = script_map[method]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    bucket_url = os.getenv('S3_BUCKET_URL')
    if not bucket_url:
        logger.error("S3_BUCKET_URL environment variable is not set")
        return False
    
    scripts_to_transfer = [
        (python_script, 'visualisation'),
        (slurm_script, 'slurm'),
        ('download_from_s3.sh', 'download')
    ]
    
    for script_name, _ in scripts_to_transfer:
        script_path = os.path.join(project_root, 'scripts', script_name)
        if not os.path.exists(script_path):
            logger.error(f"Script not found at {script_path}")
            return False
    
    try:
        ssh = SSHClient()
        ssh.connect()
        
        for script_name, script_type in scripts_to_transfer:
            local_path = os.path.join(project_root, 'scripts', script_name)
            remote_path = os.path.join(ssh.work_dir, script_name)
            
            if script_name == 'download_from_s3.sh':
                with open(local_path, 'r') as f:
                    content = f.read()
                content = content.replace(
                    'BUCKET_URL=${S3_BUCKET_URL:-""}',
                    f'BUCKET_URL="{bucket_url}"'
                )
                temp_path = local_path + '.tmp'
                with open(temp_path, 'w') as f:
                    f.write(content)
                ssh.upload_file(temp_path, remote_path)
                os.remove(temp_path)
            else:
                ssh.upload_file(local_path, remote_path)
            
            ssh.execute_command(f"chmod +x {remote_path}")
        
        # TODO: Write a script that is going to send the reduced dataset to the chroma DB. This script will be sent and exececuted on the HPC server.
        if dataset_filename:
            download_script_path = os.path.join(ssh.work_dir, 'download_from_s3.sh')
            work_folder = os.getenv('HPC_WORK_FOLDER')
            if not work_folder:
                logger.error("HPC_WORK_FOLDER environment variable is not set")
                return False
                
            exit_code, stdout, stderr = ssh.execute_command(
                f"HPC_WORK_FOLDER={work_folder} {download_script_path} {dataset_filename}"
            )
            if exit_code != 0:
                logger.error(f"Failed to download dataset: {stderr}")
                return False
            logger.info(f"Successfully downloaded dataset: {dataset_filename}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to transfer scripts to HPC: {str(e)}")
        return False
    finally:
        if 'ssh' in locals():
            ssh.disconnect()


def apply_dimensionality_reduction(embeddings: np.ndarray, method: str, params: Dict[str, int | float], transfer_to_hpc_server: bool = False, dataset_filename: str = None) -> np.ndarray:
    """
    Apply dimensionality reduction to embedding vectors using the specified method.

    Parameters:
    embeddings : np.ndarray
        The high-dimensional embedding vectors to reduce.
    method : str
        The dimensionality reduction method to use. Should be one of:
        'UMAP', 't-SNE', 'PaCMAP', 'TriMAP'.
    params : Dict[str, int | float]
        Parameters for the dimensionality reduction method.
    transfer_to_hpc_server : bool, optional
        If True, will transfer the visualization script to HPC server.
        Default is False.
    dataset_filename : str, optional
        Name of the dataset file in S3 to download on the HPC server.
        Required if transfer_to_hpc_server is True.

    Returns:
    np.ndarray
        The reduced embeddings with shape (n_samples, n_components).
    """
    if transfer_to_hpc_server:
        if not dataset_filename:
            st.error("Dataset filename is required when transferring to HPC server")
            return None
            
        status_text = st.empty()
        status_text.text(f"Transferring {method} scripts to HPC server...")
        if transfer_script_to_hpc(method, dataset_filename):
            status_text.text("Scripts transfer completed successfully!")
        else:
            st.error(f"Failed to transfer {method} scripts to HPC server")
            return None
        status_text.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if method == "UMAP":
            status_text.text("Initializing UMAP...")
            progress_bar.progress(10)
            reducer = umap.UMAP(
                n_neighbors=params["n_neighbors"],
                min_dist=params["min_dist"],
                n_components=params["n_components"],
            )
            status_text.text("Computing UMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "t-SNE":
            status_text.text("Initializing t-SNE...")
            progress_bar.progress(10)
            reducer = TSNE(
                n_components=params["n_components"],
                perplexity=params["perplexity"],
                max_iter=params["max_iter"],
            )
            status_text.text("Computing t-SNE projection (this may take a while)...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "PaCMAP":
            status_text.text("Initializing PaCMAP...")
            progress_bar.progress(10)
            reducer = pacmap.PaCMAP(
                n_neighbors=params["n_neighbors"], n_components=params["n_components"]
            )
            status_text.text("Computing PaCMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "TriMAP":
            status_text.text("Initializing TriMAP...")
            progress_bar.progress(10)
            reducer = trimap.TRIMAP(n_dims=params["n_components"], n_inliers=params["n_neighbors"])
            status_text.text("Computing TriMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        else:
            st.error(f"Unsupported dimensionality reduction method: {method}")
            return None

        status_text.text("Dimensionality reduction completed!")
        return reduced

    finally:
        def cleanup():
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        cleanup()


def save_reduction_results(
    db: VectorDB,
    reduced_embeddings: np.ndarray,
    labels: list[str],
    collection_name: str,
    type="reduced",
    method: str = None,
    params: dict = None,
) -> None:
    """
    Save dimensionality reduction results to the vector database.
    """
    metadata = {"type": type}
    if params is not None:
        metadata["params"] = json.dumps(params)
    if method is not None:
        metadata["method"] = method

    db.add_collection(collection_name, metadata=metadata)

    metadatas = [{"label": label} for label in labels]

    db.add_reduced_to_collection(collection_name, list(reduced_embeddings), metadata=metadatas)


def load_reduction_results(db: VectorDB, collection_name: str, include=["embeddings"]) -> tuple[GetResult, Collection]:
    """
    Load saved dimensionality reduction results from the vector database.

    Parameters:
    db : VectorDB
        The vector database instance to load the data from.
    collection_name : str
        The name of the collection to load.
    include : List[str], optional
        List of fields to include in the result (default is ["embeddings"]).

    Returns:
    Tuple[GetResult, Collection]
        A tuple containing:
        - reduction_results (GetResult): The reduced embeddings and associated metadata.
        - collection (Collection): The collection wit
    """

    reduction_results = db.get_all_items_from_collection(collection_name, include=include)
    collection = db.get_collection(collection_name)
    return reduction_results, collection
