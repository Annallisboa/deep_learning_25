# Parte do Streamlit para visualização interativa
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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


def main():
    st.title("Análise de Clusters - SOM")
    st.write("Visualização interativa dos clusters formados pelo Self-Organizing Map")

    # Carregar os dados
    try:
        df = pd.read_csv('imo.csv')
        st.success("Dados carregados com sucesso!")
        st.write(f"Shape do dataset: {df.shape}")
    except FileNotFoundError:
        st.error("Arquivo 'imo.csv' não encontrado! Verifique se o arquivo está no diretório correto.")
        return
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return

    # Processar dados
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Não há colunas numéricas no dataset!")
        return

    X = df[numeric_cols].values

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar com parâmetros
    st.sidebar.header("Parâmetros do SOM")
    som_size = st.sidebar.slider("Tamanho do SOM", 3, 10, 5)
    lr = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.3)
    max_epochs = st.sidebar.slider("Número de Épocas", 100, 1000, 700)
    sigma = st.sidebar.slider("Sigma", 0.1, 2.0, 1.0)

    # Botão para treinar o SOM
    if st.sidebar.button("Treinar SOM"):
        with st.spinner("Treinando o SOM... Isso pode levar alguns instantes."):
            # Criar e treinar o SOM
            input_dim = X_scaled.shape[1]
            som = SOM(som_size, input_dim, lr, sigma, random_seed=42)
            som.train(X_scaled, max_epochs)

            # Fazer previsões
            labels = som.predict(X_scaled)
            df['SOM_Cluster'] = labels

            # Salvar o modelo treinado na session state
            st.session_state.som_trained = True
            st.session_state.df_clusterizado = df
            st.session_state.labels = labels

    # Verificar se o SOM foi treinado
    if 'som_trained' in st.session_state and st.session_state.som_trained:
        df = st.session_state.df_clusterizado
        labels = st.session_state.labels

        st.subheader("Distribuição dos Clusters")
        fig, ax = plt.subplots()
        df['SOM_Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title('Distribuição de Amostras por Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Número de Amostras')
        st.pyplot(fig)

        # Selecionar variável para análise
        st.subheader("Análise por Variável")
        variavel = st.selectbox("Selecione a variável para análise:", numeric_cols)

        # Boxplot por cluster
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df.boxplot(column=variavel, by='SOM_Cluster', ax=ax2)
        plt.title(f'Distribuição de {variavel} por Cluster')
        plt.suptitle('')  # Remove título automático
        st.pyplot(fig2)

        # Estatísticas descritivas
        st.subheader("Estatísticas por Cluster")
        st.dataframe(df.groupby('SOM_Cluster')[variavel].describe())

        # Matriz U
        st.subheader("Matriz U - Visualização do SOM")
        som = SOM(som_size, X_scaled.shape[1], lr, sigma, 42)
        umatrix = som.get_umatrix()

        fig3, ax3 = plt.subplots()
        im = ax3.imshow(umatrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Matriz U - Distâncias Médias entre Neurônios')
        st.pyplot(fig3)

        # Download dos resultados
        st.subheader("Exportar Resultados")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar dados clusterizados",
            data=csv,
            file_name="dados_clusterizados_som.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Configure os parâmetros e clique em 'Treinar SOM' para iniciar a análise.")


if __name__ == "__main__":
    main()