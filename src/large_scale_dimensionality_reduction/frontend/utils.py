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
import os

from large_scale_dimensionality_reduction.vector_db import VectorDB
from large_scale_dimensionality_reduction.embeddings import Embeddings
from large_scale_dimensionality_reduction.utils import S3Client, DatasetDB, SSHClient, setup_logger
from large_scale_dimensionality_reduction.utils.config import cfg

logger = setup_logger("dim_reduction-logger")

def create_embeddings(embeddings_instance: Embeddings, uploaded_file, label_column: str = "label", description: str = None) -> Tuple[str, str] | None:
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
    Tuple[str, str] or None
        A tuple containing (collection_name, embeddings_key) if successful, None if there was an error.
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

        s3_client = S3Client()
        
        raw_data_key = None
        embeddings_key = None
        
        try:
            raw_data_key = s3_client.upload_dataframe(
                df=df,
                filename=uploaded_file.name,
                prefix="raw_data"
            )
            st.success(f"Dataset uploaded to S3: {raw_data_key}")
        except Exception as e:
            st.error(f"Failed to upload dataset to S3: {str(e)}")
            return None

        if raw_data_key:
            try:
                db = DatasetDB()
                db.add_dataset(
                    name=uploaded_file.name,
                    s3_key=raw_data_key,
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
            
            try:
                db = VectorDB()
                embeddings_result = db.get_all_items_from_collection(collection_name, include=["embeddings"])
                embeddings = np.array(embeddings_result["embeddings"])
                
                embeddings_df = pd.DataFrame(embeddings)
                embeddings_df['label'] = labels
                
                embeddings_key = s3_client.upload_dataframe(
                    df=embeddings_df,
                    filename=f"{collection_name}_embeddings.csv",
                    prefix="embeddings"
                )
                st.success(f"Embeddings saved to S3: {embeddings_key}")
                
                db_instance = DatasetDB()
                dataset_info = db_instance.get_dataset_by_collection_name(collection_name)
                if dataset_info:
                    db_instance.update_dataset(
                        dataset_info['id'],
                        embeddings_key=embeddings_key
                    )
                
            except Exception as e:
                st.error(f"Failed to save embeddings to S3: {str(e)}")
                if embeddings_key is None:
                    return None
                
        except Exception as e:
            st.error(f"Failed to store embeddings in ChromaDB: {str(e)}")
            if raw_data_key:
                try:
                    db = DatasetDB()
                    db.delete_dataset(db.get_dataset_by_s3_key(raw_data_key)["id"])
                    
                    s3_client = S3Client()
                    s3_client.delete_object(raw_data_key)
                    st.info(f"Cleaned up: removed dataset from S3 and local database")
                except Exception as cleanup_error:
                    st.warning(f"Failed to clean up resources: {str(cleanup_error)}")
            return None

        return collection_name, embeddings_key

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


def transfer_script_to_hpc(method: str, dataset_filename: str = None, params: Dict[str, int | float] = None) -> bool:
    """
    Transfer the appropriate visualization script, its slurm job script, and download script to HPC server.
    Also submits a SLURM job if parameters are provided.
    
    Args:
        method: The dimensionality reduction method ('UMAP', 't-SNE', 'PaCMAP', or 'TriMAP')
        dataset_filename: Optional name of the dataset file in S3 to download
        params: Optional parameters for the dimensionality reduction method
        
    Returns:
        bool: True if transfer and job submission were successful, False otherwise
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
        ('download_from_s3.sh', 'download'),
        ('send_to_chroma.py', 'chroma')
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
                ssh.execute_command(f"chmod +x {remote_path}")
                os.remove(temp_path)
            elif script_name == 'send_to_chroma.py':
                ssh.upload_file(local_path, remote_path)
                ssh.execute_command(f"chmod +x {remote_path}")
            else:
                ssh.upload_file(local_path, remote_path)
                ssh.execute_command(f"chmod +x {remote_path}")
        
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
            
            if params is not None:
                hpc_filename = dataset_filename.replace('embeddings/', '')
                input_file = os.path.join(work_folder, hpc_filename)
                output_file = os.path.join(work_folder, f"{hpc_filename[:-4]}_reduced.csv")
                
                slurm_script_path = os.path.join(ssh.work_dir, slurm_script)
                with open(os.path.join(project_root, 'scripts', slurm_script), 'r') as f:
                    slurm_content = f.read()
                
                if method == 'UMAP':
                    slurm_content = slurm_content.replace('{{n_components}}', str(params['n_components']))
                    slurm_content = slurm_content.replace('{{n_neighbors}}', str(params['n_neighbors']))
                    slurm_content = slurm_content.replace('{{min_dist}}', str(params['min_dist']))
                elif method == 't-SNE':
                    slurm_content = slurm_content.replace('{{n_components}}', str(params['n_components']))
                    slurm_content = slurm_content.replace('{{perplexity}}', str(params['perplexity']))
                    slurm_content = slurm_content.replace('{{max_iter}}', str(params['max_iter']))
                elif method == 'PaCMAP':
                    slurm_content = slurm_content.replace('{{n_components}}', str(params['n_components']))
                    slurm_content = slurm_content.replace('{{n_neighbors}}', str(params['n_neighbors']))
                elif method == 'TriMAP':
                    slurm_content = slurm_content.replace('{{n_components}}', str(params['n_components']))
                    slurm_content = slurm_content.replace('{{n_neighbors}}', str(params['n_neighbors']))
                
                slurm_content = slurm_content.replace('input.csv', input_file)
                slurm_content = slurm_content.replace('output.csv', output_file)
                
                collection_name = f"{hpc_filename[:-4]}_reduced_{method.lower()}"
                if method == 'UMAP':
                    collection_name += f"_n{params['n_neighbors']}_d{params['min_dist']}"
                elif method == 't-SNE':
                    collection_name += f"_p{params['perplexity']}_i{params['max_iter']}"
                elif method == 'PaCMAP':
                    collection_name += f"_n{params['n_neighbors']}"
                elif method == 'TriMAP':
                    collection_name += f"_n{params['n_neighbors']}"
                
                slurm_content += f"\npython3 {os.path.join(ssh.work_dir, 'send_to_chroma.py')} {output_file} {collection_name} {method} '{json.dumps(params)}' {cfg.CHROMA_HOST} {cfg.CHROMA_PORT}\n"
                
                temp_slurm_path = os.path.join(ssh.work_dir, f"temp_{slurm_script}")
                ssh.execute_command(f"cat > {temp_slurm_path} << 'EOL'\n{slurm_content}\nEOL")
                ssh.execute_command(f"chmod +x {temp_slurm_path}")
                
                logger.info(f"Submitting SLURM job for {method}...")
                exit_code, stdout, stderr = ssh.execute_command(f"sbatch {temp_slurm_path}")
                if exit_code != 0:
                    logger.error(f"Failed to submit SLURM job: {stderr}")
                    return False
                
                job_id = stdout.strip().split()[-1]
                logger.info(f"SLURM job submitted with ID: {job_id}")
                
                ssh.execute_command(f"rm {temp_slurm_path}")
                return True
        
        return True
    except Exception as e:
        logger.error(f"Failed to transfer scripts to HPC: {str(e)}")
        return False
    finally:
        if 'ssh' in locals():
            ssh.disconnect()


def apply_dimensionality_reduction(method: str, params: Dict[str, int | float], dataset_filename: str = None):
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
        Default is False.
    dataset_filename : str, optional
        Name of the dataset file in S3 to download on the HPC server.

    Returns:
    np.ndarray
        The reduced embeddings with shape (n_samples, n_components).
    """
    
    if not dataset_filename:
        st.error("Dataset filename is required when transferring to HPC server")
        return None
        
    status_text = st.empty()
    status_text.text(f"Transferring {method} scripts to HPC server and submitting job...")
    
    if transfer_script_to_hpc(method, dataset_filename, params):
        status_text.text("Job submitted successfully! The results will be available in the HPC work directory.")
    else:
        st.error(f"Failed to submit {method} job to HPC server")


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
