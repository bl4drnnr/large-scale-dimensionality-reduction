#python visualisations_script_tsne.py input.csv output.csv n_components perplexity max_iter
import pandas as pd
from sklearn.manifold import TSNE
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
n_components = int(sys.argv[3])
perplexity = float(sys.argv[4])
max_iter = int(sys.argv[5])

df = pd.read_csv(input_file)
X = df.select_dtypes(include='number')

reducer = TSNE(
    n_components=n_components,
    perplexity=perplexity,
    max_iter=max_iter
)
components = reducer.fit_transform(X)

df_reduced = pd.DataFrame(components, columns=[f"tSNE{i+1}" for i in range(n_components)])
df_reduced.to_csv(output_file, index=False) 