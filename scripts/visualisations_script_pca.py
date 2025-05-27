import pandas as pd
from sklearn.decomposition import PCA
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
n_components = int(sys.argv[3])

df = pd.read_csv(input_file)
X = df.select_dtypes(include='number')

pca = PCA(n_components=n_components)
components = pca.fit_transform(X)

df_pca = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
df_pca.to_csv(output_file, index=False)

