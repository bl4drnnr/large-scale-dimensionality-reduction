import streamlit as st
import pandas as pd
from large_scale_dimensionality_reduction.utils import DatasetDB
from large_scale_dimensionality_reduction.frontend.utils import download_dataset_from_s3
from large_scale_dimensionality_reduction.utils import S3Client

st.set_page_config(
    page_title="Datasets - Text Embedding Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Datasets")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <a href='/' target='_self' style='text-decoration: none;'>
        <button style='
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            width: 100%;
        '>
            📊 Back to Main Page
        </button>
    </a>
</div>
<div style='text-align: center; margin-bottom: 20px;'>
    <a href='/saved_reductions' target='_self' style='text-decoration: none;'>
        <button style='
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            width: 100%;
        '>
            📚 View Saved Reductions
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

db = DatasetDB()

datasets = db.get_all_datasets()

if not datasets:
    st.info("No datasets have been uploaded yet.")
else:
    st.subheader("Available Datasets")
    
    df_datasets = pd.DataFrame(datasets)
    df_datasets['uploaded_at'] = pd.to_datetime(df_datasets['uploaded_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        df_datasets[[
            'name', 'collection_name', 'label_column',
            'num_rows', 'uploaded_at', 'description'
        ]].rename(columns={
            'name': 'Filename',
            'collection_name': 'Collection',
            'label_column': 'Label Column',
            'num_rows': 'Rows',
            'uploaded_at': 'Uploaded At',
            'description': 'Description'
        }),
        use_container_width=True
    )
    
    st.subheader("Dataset Actions")
    
    selected_dataset = st.selectbox(
        "Select a dataset to download",
        options=datasets,
        format_func=lambda x: f"{x['name']} ({x['collection_name']})"
    )
    
    if selected_dataset:
        new_description = st.text_area(
            "Dataset Description",
            value=selected_dataset['description'] or "",
            key=f"desc_{selected_dataset['id']}",
            help="Add or update the description for this dataset"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📝 Update Description", use_container_width=True):
                if db.update_dataset(selected_dataset['id'], description=new_description):
                    st.success("Description updated!")
                    st.rerun()
                else:
                    st.error("Failed to update description")
        
        with col2:
            if st.button("⬇️ Download Dataset", use_container_width=True):
                with st.spinner("Downloading dataset..."):
                    df = download_dataset_from_s3(selected_dataset['s3_key'])
                    if df is not None:
                        st.download_button(
                            "📥 Click to download CSV",
                            df.to_csv(index=False).encode('utf-8'),
                            file_name=selected_dataset['name'],
                            mime='text/csv',
                            use_container_width=True
                        )
        
        with col3:
            if st.button("🗑️ Delete Dataset", type="primary", use_container_width=True):
                try:
                    s3_client = S3Client()
                    deletion_successful = True
                    
                    try:
                        s3_client.delete_object(selected_dataset['s3_key'])
                        st.success("Deleted raw data file from S3")
                    except Exception as e:
                        st.error(f"Failed to delete raw data file: {str(e)}")
                        deletion_successful = False
                    
                    if deletion_successful and selected_dataset.get('embeddings_key'):
                        try:
                            s3_client.delete_object(selected_dataset['embeddings_key'])
                            st.success("Deleted embeddings file from S3")
                        except Exception as e:
                            st.error(f"Failed to delete embeddings file: {str(e)}")
                            deletion_successful = False
                    
                    if deletion_successful:
                        if db.delete_dataset(selected_dataset['id']):
                            st.success("Dataset deleted successfully from database!")
                            st.rerun()
                        else:
                            st.error("Failed to delete dataset from database")
                    else:
                        st.error("Dataset deletion was not completed due to S3 errors")
                except Exception as e:
                    st.error(f"Error deleting dataset: {str(e)}") 
