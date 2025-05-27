# python visualisations_script_umap.py input.csv output.csv n_components n_neighbors min_dist
import pandas as pd
import umap
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
n_components = int(sys.argv[3])
n_neighbors = int(sys.argv[4])
min_dist = float(sys.argv[5])

df = pd.read_csv(input_file)
X = df.select_dtypes(include='number')

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=n_components
)
components = reducer.fit_transform(X)

df_reduced = pd.DataFrame(components, columns=[f"UMAP{i+1}" for i in range(n_components)])
df_reduced.to_csv(output_file, index=False) 