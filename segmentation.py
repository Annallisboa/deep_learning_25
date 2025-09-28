import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('imo.csv')

# Visualizar as primeiras linhas
print("Primeiras linhas do dataset:")
print(df.head())
print(f"\nDimensões do dataset: {df.shape}")

# Preparar os dados para o SOM
# Vamos usar todas as colunas numéricas, exceto possivelmente a target se existir
# Primeiro, vamos identificar as colunas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColunas numéricas: {numeric_cols}")

# Remover colunas que não devem ser usadas no clustering (como IDs ou targets)
# Vamos manter todas as colunas numéricas por enquanto
X = df[numeric_cols].values

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dados normalizados shape: {X_scaled.shape}")


# Implementação do SOM
class SOM:
    def __init__(self, som_size, input_dim, lr=0.3, sigma=1.0, random_seed=42):
        self.som_size = som_size
        self.input_dim = input_dim
        self.lr = lr
        self.sigma = sigma
        self.random_seed = random_seed

        # Inicializar os pesos aleatoriamente
        np.random.seed(random_seed)
        self.weights = np.random.rand(som_size, som_size, input_dim)

    def find_bmu(self, x):
        """Encontrar a Best Matching Unit (BMU) para um vetor de entrada x"""
        # Calcular a distância euclidiana entre x e todos os neurônios
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Encontrar o índice do neurônio com menor distância
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_weights(self, x, bmu_index, epoch, max_epochs):
        """Atualizar os pesos do SOM"""
        # Calcular a taxa de aprendizado e raio de vizinhança decrescentes
        current_lr = self.lr * (1 - epoch / max_epochs)
        current_sigma = self.sigma * (1 - epoch / max_epochs)

        # Criar grid de coordenadas
        i = np.arange(self.som_size)
        j = np.arange(self.som_size)
        ii, jj = np.meshgrid(i, j, indexing='ij')

        # Calcular distâncias no grid do SOM
        distances = np.sqrt((ii - bmu_index[0]) ** 2 + (jj - bmu_index[1]) ** 2)

        # Calcular a função de vizinhança
        neighborhood = np.exp(-distances ** 2 / (2 * current_sigma ** 2))

        # Atualizar os pesos
        for i in range(self.som_size):
            for j in range(self.som_size):
                influence = neighborhood[i, j] * current_lr
                self.weights[i, j] += influence * (x - self.weights[i, j])

    def train(self, X, max_epochs=700):
        """Treinar o SOM"""
        n_samples = X.shape[0]

        for epoch in range(max_epochs):
            # Embaralhar os dados a cada época
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x = X[idx]
                # Encontrar BMU
                bmu_index = self.find_bmu(x)
                # Atualizar pesos
                self.update_weights(x, bmu_index, epoch, max_epochs)

            if epoch % 100 == 0:
                print(f"Época {epoch}/{max_epochs} concluída")

    def predict(self, X):
        """Atribuir cada amostra a um neurônio do SOM"""
        labels = []
        for x in X:
            bmu_index = self.find_bmu(x)
            # Converter coordenadas 2D em um único label
            label = bmu_index[0] * self.som_size + bmu_index[1]
            labels.append(label)
        return np.array(labels)

    def get_umatrix(self):
        """Calcular a matriz U (Unified Distance Matrix) para visualização"""
        umatrix = np.zeros((self.som_size, self.som_size))

        for i in range(self.som_size):
            for j in range(self.som_size):
                # Calcular distâncias para os vizinhos
                distances = []
                if i > 0:
                    distances.append(np.linalg.norm(self.weights[i, j] - self.weights[i - 1, j]))
                if i < self.som_size - 1:
                    distances.append(np.linalg.norm(self.weights[i, j] - self.weights[i + 1, j]))
                if j > 0:
                    distances.append(np.linalg.norm(self.weights[i, j] - self.weights[i, j - 1]))
                if j < self.som_size - 1:
                    distances.append(np.linalg.norm(self.weights[i, j] - self.weights[i, j + 1]))

                umatrix[i, j] = np.mean(distances) if distances else 0

        return umatrix


# Criar e treinar o SOM
som_size = 5
input_dim = X_scaled.shape[1]
lr = 0.3
sigma = 1.0
max_epochs = 700
random_seed = 42

print(f"\nIniciando treinamento do SOM...")
print(f"Tamanho do SOM: {som_size}x{som_size}")
print(f"Dimensão de entrada: {input_dim}")
print(f"Taxa de aprendizado: {lr}")
print(f"Sigma: {sigma}")
print(f"Máximo de épocas: {max_epochs}")
print(f"Random seed: {random_seed}")

som = SOM(som_size, input_dim, lr, sigma, random_seed)
som.train(X_scaled, max_epochs)

print("\nTreinamento concluído!")

# Fazer previsões
labels = som.predict(X_scaled)
df['SOM_Cluster'] = labels

# Análise dos resultados
print(f"\nDistribuição dos clusters:")
cluster_counts = df['SOM_Cluster'].value_counts().sort_index()
print(cluster_counts)

# Visualizações
# plt.figure(figsize=(15, 10))

# 1. Matriz U (U-Matrix)
# plt.subplot(2, 3, 1)
# umatrix = som.get_umatrix()
# plt.imshow(umatrix, cmap='viridis', origin='lower')
# plt.colorbar(label='Distância Média')
# plt.title('U-Matrix do SOM')
# plt.xlabel('X')
# plt.ylabel('Y')

# 2. Distribuição dos clusters
# plt.subplot(2, 3, 2)
# plt.bar(cluster_counts.index, cluster_counts.values)
# plt.title('Distribuição dos Clusters')
# plt.xlabel('Cluster')
# plt.ylabel('Número de Amostras')
# plt.xticks(cluster_counts.index)

# 3. Mapa de calor dos clusters
# plt.subplot(2, 3, 3)
# cluster_map = np.zeros((som_size, som_size))
# for i in range(som_size):
#     for j in range(som_size):
#         cluster_id = i * som_size + j
#         if cluster_id in df['SOM_Cluster'].values:
#             cluster_map[i, j] = (df['SOM_Cluster'] == cluster_id).sum()
#         else:
#             cluster_map[i, j] = 0
#
# plt.imshow(cluster_map, cmap='YlOrRd', origin='lower')
# plt.colorbar(label='Número de Amostras')
# plt.title('Mapa de Calor dos Clusters')
# plt.xlabel('X')
# plt.ylabel('Y')

# Adicionar anotações com contagens
# for i in range(som_size):
#     for j in range(som_size):
#         plt.text(j, i, f'{int(cluster_map[i, j])}',
#                  ha='center', va='center', color='black', fontweight='bold')

# 4. Análise de variáveis por cluster (exemplo com algumas variáveis importantes)
# Selecionar algumas variáveis para análise
# if 'Spent' in df.columns:
#     plt.subplot(2, 3, 4)
#     sns.boxplot(data=df, x='SOM_Cluster', y='Spent')
#     plt.title('Gasto por Cluster')
#     plt.xticks(rotation=45)
#
# if 'Income' in df.columns:
#     plt.subplot(2, 3, 5)
#     sns.boxplot(data=df, x='SOM_Cluster', y='Income')
#     plt.title('Renda por Cluster')
#     plt.xticks(rotation=45)
#
# if 'Age_on_2024' in df.columns:
#     plt.subplot(2, 3, 6)
#     sns.boxplot(data=df, x='SOM_Cluster', y='Age_on_2024')
#     plt.title('Idade por Cluster')
#     plt.xticks(rotation=45)
#
# plt.tight_layout()
# plt.show()

# Análise descritiva por cluster
print("\nAnálise descritiva por cluster:")
numeric_columns_for_analysis = ['Income', 'Spent', 'Age_on_2024', 'MntWines', 'MntMeatProducts']

for col in numeric_columns_for_analysis:
    if col in df.columns:
        print(f"\n{col} por cluster:")
        cluster_stats = df.groupby('SOM_Cluster')[col].describe()
        print(cluster_stats)

# Salvar resultados
# output_filename = 'clustered_data_som_results.csv0'
# df.to_excel(output_filename, index=False)
# print(f"\nResultados salvos em: {output_filename}")

# Resumo final
# print("\n=== RESUMO DO SOM ===")
# print(f"Total de clusters formados: {len(cluster_counts)}")
# print(f"Tamanho do menor cluster: {cluster_counts.min()} amostras")
# print(f"Tamanho do maior cluster: {cluster_counts.max()} amostras")
# print(f"Cluster mais populoso: {cluster_counts.idxmax()} com {cluster_counts.max()} amostras")

