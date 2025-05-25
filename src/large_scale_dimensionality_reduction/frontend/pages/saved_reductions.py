import streamlit as st
from pathlib import Path
from large_scale_dimensionality_reduction.frontend.utils import load_reduction_results
from large_scale_dimensionality_reduction.frontend.visualizations import plot_reduced_embeddings

st.set_page_config(
    page_title="Saved Reductions",
    page_icon="📚",
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

st.title("Saved Reductions")
st.markdown("""
This page allows you to view and interact with your saved dimensionality reductions.
Select a saved reduction from the list below to visualize it.
""")

saved_reductions_dir = Path("saved_reductions")
saved_reductions = list(saved_reductions_dir.glob("*.pkl")) if saved_reductions_dir.exists() else []

if not saved_reductions:
    st.info("No saved reductions found. Go to the main page to create and save reductions.")
    st.stop()

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
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("Saved Reductions")
selected_reduction = st.sidebar.selectbox(
    "Select a reduction to view",
    options=[f.stem for f in saved_reductions],
    help="Choose a saved reduction to visualize"
)

if selected_reduction:
    try:
        reduced_embeddings, labels, method, params = load_reduction_results(selected_reduction)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Reduction Details")
        st.sidebar.write(f"**Method:** {method}")
        st.sidebar.write("**Parameters:**")
        for param, value in params.items():
            st.sidebar.write(f"- {param}: {value}")
        
        tab2D, tab3D = st.tabs(["2D Visualization", "3D Visualization"])
        
        with tab2D:
            fig = plot_reduced_embeddings(
                reduced_embeddings,
                labels,
                method,
                type="2D"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button(
                label="Download 2D visualization as HTML",
                data=fig.to_html(include_plotlyjs="cdn"),
                file_name=f"{selected_reduction}_2d.html",
                mime="text/html"
            )
        
        with tab3D:
            fig3D = plot_reduced_embeddings(
                reduced_embeddings,
                labels,
                method,
                type="3D"
            )
            st.plotly_chart(fig3D, use_container_width=True)
            
            st.download_button(
                label="Download 3D visualization as HTML",
                data=fig3D.to_html(include_plotlyjs="cdn"),
                file_name=f"{selected_reduction}_3d.html",
                mime="text/html"
            )
        
        st.markdown("---")
        st.subheader("Reduction Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of points", len(reduced_embeddings))
        with col2:
            st.metric("Number of unique labels", len(set(labels)))
        with col3:
            st.metric("Method", method)
        
    except Exception as e:
        st.error(f"Error loading reduction: {str(e)}") 