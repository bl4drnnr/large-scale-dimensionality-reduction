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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Dataset"):
                with st.spinner("Downloading dataset..."):
                    df = download_dataset_from_s3(selected_dataset['s3_key'])
                    if df is not None:
                        st.download_button(
                            "Click to download CSV",
                            df.to_csv(index=False).encode('utf-8'),
                            file_name=selected_dataset['name'],
                            mime='text/csv'
                        )
        
        with col2:
            new_description = st.text_area(
                "Update description",
                value=selected_dataset['description'] or "",
                key=f"desc_{selected_dataset['id']}"
            )
            if st.button("Update Description"):
                if db.update_dataset_description(selected_dataset['id'], new_description):
                    st.success("Description updated!")
                    st.rerun()
                else:
                    st.error("Failed to update description")
        
        if st.button("Delete Dataset", type="primary"):
            try:
                s3_client = S3Client()
                s3_client.delete_object(selected_dataset['s3_key'])
                
                if db.delete_dataset(selected_dataset['id']):
                    st.success("Dataset deleted successfully from both S3 and database!")
                    st.rerun()
                else:
                    st.error("Failed to delete dataset from database")
            except Exception as e:
                st.error(f"Error deleting dataset: {str(e)}") 
