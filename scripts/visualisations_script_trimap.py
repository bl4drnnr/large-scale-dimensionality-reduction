# python visualisations_script_trimap.py input.csv output.csv n_components n_neighbors
import pandas as pd
import trimap
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
n_components = int(sys.argv[3])
n_neighbors = int(sys.argv[4])

df = pd.read_csv(input_file)
X = df.select_dtypes(include='number')

reducer = trimap.TRIMAP(
    n_dims=n_components,
    n_inliers=n_neighbors
)
components = reducer.fit_transform(X)

df_reduced = pd.DataFrame(components, columns=[f"TriMAP{i+1}" for i in range(n_components)])
df_reduced.to_csv(output_file, index=False) 