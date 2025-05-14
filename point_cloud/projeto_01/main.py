"""
PointNet Implementation for 3D Point Cloud Classification
========================================================

Este script implementa a arquitetura PointNet para classificação de nuvens de pontos 3D
usando o dataset ModelNet10. A implementação está otimizada para execução em CPU.

Principais características:
- Classificação de 10 classes de objetos 3D
- Transformação Spatial Transformer Networks (T-Net) para invariância a transformações
- Otimizações específicas para CPU (AMD Ryzen)
- Visualização de resultados e métricas de treinamento

Referência: PointNet - Deep Learning on Point Sets for 3D Classification and Segmentation
            Qi et al., CVPR 2017
"""

import sys
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from utils import download_modelnet

# ======================================================================
# CONFIGURAÇÃO DO AMBIENTE E HARDWARE
# ======================================================================

def configure_cpu_optimizations():
    """
    Configura TensorFlow para execução otimizada em CPU.
    
    Esta função:
    - Força o uso exclusivo da CPU (desabilitando GPU)
    - Configura paralelismo inter e intra operações para usar todos os cores
    - Define semente para reprodutibilidade
    """
    print("Configurando para CPU (AMD Ryzen)...")
    tf.config.set_visible_devices([], 'GPU')  # Força uso de CPU
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Usa todos os cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Otimizada para multi-core
    tf.random.set_seed(1234)

# Executa configurações de CPU
configure_cpu_optimizations()

# Configuração de caminhos
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ======================================================================
# DOWNLOAD E CARREGAMENTO DO DATASET
# ======================================================================

# Download do dataset ModelNet
DATA_DIR = download_modelnet.download()
print(f"Dataset disponível em: {DATA_DIR}")

def visualize_sample_mesh():
    """
    Carrega e visualiza uma amostra do dataset para verificação.
    
    Tenta carregar um modelo 3D de cadeira e visualizar seus pontos
    amostrados em um gráfico 3D.
    """
    try:
        mesh_path = os.path.join(DATA_DIR, "chair/train/chair_0002.off")
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
            print(f"Mesh carregada: {mesh}")
            
            # Amostra pontos da malha 3D
            points = mesh.sample(2048)
            
            # Visualiza os pontos em um gráfico 3D
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            ax.set_axis_off()
            plt.show()
        else:
            print(f"Arquivo não encontrado: {mesh_path}")
            print(f"Conteúdo de DATA_DIR: {os.listdir(DATA_DIR)}")
            
            chair_paths = glob.glob(os.path.join(DATA_DIR, "*chair*"))
            print(f"Possíveis caminhos para cadeiras: {chair_paths}")
    except Exception as e:
        print(f"Erro ao carregar ou visualizar o modelo: {e}")

# Visualiza uma amostra do dataset
visualize_sample_mesh()

# ======================================================================
# PROCESSAMENTO DO DATASET
# ======================================================================

def parse_dataset(num_points=2048):
    """
    Processa o dataset ModelNet e prepara os dados para treinamento.
    
    Args:
        num_points (int): Número de pontos a serem amostrados de cada mesh
        
    Returns:
        tuple: Contém:
            - train_points (np.array): Pontos de treinamento (N, num_points, 3)
            - test_points (np.array): Pontos de teste (N, num_points, 3)
            - train_labels (np.array): Labels de treinamento
            - test_labels (np.array): Labels de teste
            - class_map (dict): Mapeamento entre índices e nomes das classes
    """
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    
    # Encontra todas as pastas de classes (exclui README)
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
        
        # Arquivos de treinamento e teste
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        # Processa arquivos de treinamento
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        # Processa arquivos de teste
        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

# ======================================================================
# HIPERPARÂMETROS
# ======================================================================

NUM_POINTS = 1800      # Número de pontos por nuvem
NUM_CLASSES = 10       # Número de classes no ModelNet10
BATCH_SIZE = 16        # Tamanho do batch (reduzido para CPU)

# Carrega e processa o dataset
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

# ======================================================================
# DATA AUGMENTATION E PIPELINE DE DADOS
# ======================================================================

def augment(points, label):
    """
    Aplica data augmentation nos pontos.
    
    Args:
        points (tf.Tensor): Nuvem de pontos (num_points, 3)
        label (tf.Tensor): Label da classe
        
    Returns:
        tuple: Pontos aumentados e label
        
    Transformações aplicadas:
    - Adiciona ruído gaussiano pequeno
    - Embaralha a ordem dos pontos (PointNet é invariante à ordem)
    """
    # Adiciona ruído gaussiano pequeno
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # Embaralha pontos (invariância à ordem)
    points = tf.random.shuffle(points)
    return points, label

def create_optimized_dataset(points, labels, is_training=True):
    """
    Cria pipeline de dados otimizado para CPU.
    
    Args:
        points (np.array): Array de nuvens de pontos
        labels (np.array): Array de labels
        is_training (bool): Se é dataset de treinamento (aplica augmentation)
        
    Returns:
        tf.data.Dataset: Dataset otimizado
    """
    dataset = tf.data.Dataset.from_tensor_slices((points, labels))
    
    if is_training:
        dataset = dataset.shuffle(len(points)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Cria datasets de treinamento e teste
train_dataset = create_optimized_dataset(train_points, train_labels, is_training=True)
test_dataset = create_optimized_dataset(test_points, test_labels, is_training=False)

# ======================================================================
# BUILDING BLOCKS DA ARQUITETURA POINTNET
# ======================================================================

def conv_bn(x, filters):
    """
    Bloco convolucional 1D + Batch Normalization + ReLU.
    
    Args:
        x (tf.Tensor): Input tensor
        filters (int): Número de filtros convolucionais
        
    Returns:
        tf.Tensor: Output após conv1d -> bn -> relu
    """
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    """
    Bloco dense + Batch Normalization + ReLU.
    
    Args:
        x (tf.Tensor): Input tensor
        filters (int): Número de neurônios
        
    Returns:
        tf.Tensor: Output após dense -> bn -> relu
    """
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

# ======================================================================
# SPATIAL TRANSFORMER NETWORK (T-NET)
# ======================================================================

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    Regularizador ortogonal para T-Net.
    
    Penaliza desvios da ortogonalidade na matriz de transformação
    para manter estabilidade numérica e validez geométrica.
    
    Args:
        num_features (int): Dimensão da matriz de transformação
        l2reg (float): Peso da regularização
    """
    
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        """
        Calcula a penalidade ortogonal.
        
        Args:
            x (tf.Tensor): Matriz de transformação
            
        Returns:
            tf.Tensor: Valor da penalidade
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    def get_config(self):
        """
        Retorna a configuração do regularizador para serialização.
        
        Returns:
            dict: Configuração com parâmetros do regularizador
        """
        return {
            'num_features': self.num_features,
            'l2reg': self.l2reg
        }
    
    @classmethod
    def from_config(cls, config):
        """
        Cria instância do regularizador a partir da configuração.
        
        Args:
            config (dict): Configuração salva
            
        Returns:
            OrthogonalRegularizer: Nova instância do regularizador
        """
        return cls(**config)

def tnet(inputs, num_features):
    """
    Implementa o Transformation Network (T-Net) do PointNet.
    
    O T-Net aprende uma transformação matricial para tornar a rede
    invariante a certas transformações geométricas (rotação, translação).
    
    Args:
        inputs (tf.Tensor): Nuvem de pontos de entrada (batch_size, num_points, num_features)
        num_features (int): Dimensão dos features (3 para pontos 3D, 64 para features)
        
    Returns:
        tf.Tensor: Pontos transformados pela matriz aprendida
    """
    # Inicializa bias como matriz identidade (transformação neutra)
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    # Regularizador para manter ortogonalidade
    reg = OrthogonalRegularizer(num_features)

    # Feature extraction através de convoluções 1D
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64) 
    x = conv_bn(x, 512)
    
    # Global max pooling para obter features globais
    x = layers.GlobalMaxPooling1D()(x)
    
    # MLPs para predizer matriz de transformação
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    
    # Camada final para matriz de transformação
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    
    # Reshape para matriz e aplica transformação
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

# ======================================================================
# ARQUITETURA POINTNET COMPLETA
# ======================================================================

def build_pointnet_model():
    """
    Constrói o modelo PointNet completo.
    
    Arquitetura:
    1. T-Net espacial (3x3) para invariância a transformações rígidas
    2. Convoluções 1D para extrair features locais
    3. T-Net de features (64x64) para alinhamento de features
    4. Mais convoluções para features de alta dimensão
    5. Global max pooling para aggregação permutation-invariant
    6. MLPs para classificação final
    
    Returns:
        keras.Model: Modelo PointNet compilado
    """
    inputs = keras.Input(shape=(NUM_POINTS, 3))

    # T-Net espacial (aprende transformação 3x3)
    x = tnet(inputs, 3)
    
    # Convoluções para features locais
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    
    # T-Net de features (aprende transformação 64x64)
    x = tnet(x, 32)
    
    # Mais convoluções para features de alta dimensão
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    
    # Global max pooling - crucial para invariância à permutação
    x = layers.GlobalMaxPooling1D()(x)
    
    # MLPs finais para classificação
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)  # Regularização
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)  # Regularização

    # Camada de classificação
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

# Construir modelo
model = build_pointnet_model()
model.summary()

# ======================================================================
# COMPILAÇÃO E TREINAMENTO
# ======================================================================

# Optimizer otimizado para CPU
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["sparse_categorical_accuracy"],
)

def create_callbacks():
    """
    Cria callbacks para treinamento.
    
    Returns:
        list: Lista de callbacks incluindo:
            - ReduceLROnPlateau: Reduz learning rate quando validação estagnar
            - EarlyStopping: Para treinamento quando não há melhoria
    """
    callbacks = []
    
    # Reduz learning rate quando validação estagnar
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    )
    
    # Para treinamento antecipado se não houver melhoria
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    return callbacks

# Criar callbacks
callbacks = create_callbacks()

print("Iniciando treinamento (CPU)...")
print(f"Usando {os.cpu_count()} cores do processador")

# Treinamento com menos epochs para CPU
history = model.fit(
    train_dataset, 
    epochs=10,  # Reduzido para CPU
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# ======================================================================
# VISUALIZAÇÃO DOS RESULTADOS
# ======================================================================

def plot_training_history(history):
    """
    Plota o histórico de treinamento.
    
    Args:
        history: Objeto History retornado pelo model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot da loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot da accuracy
    ax2.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Plot do histórico de treinamento
if 'history' in locals():
    plot_training_history(history)

# ======================================================================
# AVALIAÇÃO E VISUALIZAÇÃO DE PREDIÇÕES
# ======================================================================

def visualize_predictions(model, test_dataset, class_map, num_samples=8):
    """
    Visualiza predições do modelo em amostras de teste.
    
    Args:
        model: Modelo treinado
        test_dataset: Dataset de teste
        class_map: Mapeamento de classes
        num_samples: Número de amostras a visualizar
    """
    # Pega uma batch de teste
    data = test_dataset.take(1)
    points, labels = list(data)[0]
    points = points[:num_samples, ...]
    labels = labels[:num_samples, ...]

    # Faz predições
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # Visualização dos resultados
    fig = plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                class_map[preds[i].numpy()], class_map[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()
    plt.show()

# Visualiza predições
visualize_predictions(model, test_dataset, CLASS_MAP)

# ======================================================================
# TESTE COM MODELO PERSONALIZADO
# ======================================================================

def test_custom_model(model, model_path, class_map, num_points):
    """
    Testa o modelo em um arquivo 3D personalizado.
    
    Args:
        model: Modelo treinado
        model_path: Caminho para o arquivo 3D
        class_map: Mapeamento de classes
        num_points: Número de pontos a amostrar
    """
    project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    custom_model_path = os.path.join(project_root, model_path)

    print(f"Diretório atual (raiz do projeto): {project_root}")
    print(f"Caminho completo do modelo: {custom_model_path}")

    file_exists = os.path.exists(custom_model_path)
    print(f"O arquivo existe? {file_exists}")

    if file_exists:
        # Carrega o modelo 3D
        mesh_test = trimesh.load(custom_model_path)

        # Lida com scenes (múltiplas geometrias)
        if isinstance(mesh_test, trimesh.Scene):
            mesh_test = trimesh.util.concatenate(
                [g for g in mesh_test.geometry.values()]
            )

        # Amostra pontos
        points = mesh_test.sample(num_points)

        # Visualiza pontos originais
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='skyblue', s=1)
        ax.view_init(elev=30, azim=45)
        ax.set_axis_off()
        plt.show()

        # Normalização (importante para PointNet)
        points = tf.constant(points, dtype=tf.float32)
        points = points - tf.reduce_mean(points, axis=0)  # Centraliza
        points = points / tf.reduce_max(tf.norm(points, axis=1))  # Normaliza escala
        
        # Adiciona dimensão de batch
        points_tf = tf.expand_dims(points, 0)

        # Faz predição
        pred = model.predict(points_tf)
        pred_label = tf.argmax(pred, axis=-1).numpy()[0]

        points = points.numpy()

        # Visualiza resultado com predição
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='skyblue', s=1)
        
        pred_class = class_map[pred_label]
        ax.set_title(f"Resultado da Classificação: {pred_class}")
        ax.set_axis_off()
        plt.show()

        print(f"Classificação final: {pred_class} (Classe {pred_label})")
        
        # Mostra probabilidades para cada classe
        print("\nProbabilidades por classe:")
        for i, prob in enumerate(pred[0]):
            print(f"{class_map[i]}: {prob:.3f}")

# Teste com modelo personalizado
test_custom_model(model, "monitor.obj", CLASS_MAP, NUM_POINTS)

# ======================================================================
# INFORMAÇÕES FINAIS DE PERFORMANCE
# ======================================================================

print(f"\n=== Informações de Performance ===")
print(f"Processo executado em CPU")
print(f"Processador: AMD Ryzen")
print(f"Cores disponíveis: {os.cpu_count()}")
print(f"Número de pontos por nuvem: {NUM_POINTS}")
print(f"Número de classes: {NUM_CLASSES}")
print(f"Batch size: {BATCH_SIZE}")
print("Para melhor performance, considere usar Google Colab ou serviços cloud com GPU")

# ======================================================================
# SALVAMENTO DO MODELO (OPCIONAL)
# ======================================================================

def save_model(model, save_path="pointnet_model.keras"):
    """
    Salva o modelo treinado no formato nativo do Keras.
    
    Args:
        model: Modelo a ser salvo
        save_path: Caminho para salvar o modelo (formato .keras recomendado)
    """
    try:
        model.save(save_path)
        print(f"Modelo salvo com sucesso em: {save_path}")
        print(f"Tamanho do arquivo: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Erro ao salvar modelo: {e}")
        # Fallback: salvar somente os pesos
        weights_path = save_path.replace('.keras', '_weights.h5')
        model.save_weights(weights_path)
        print(f"Salvos apenas os pesos em: {weights_path}")

def load_model(model_path="pointnet_model.keras"):
    """
    Carrega um modelo PointNet salvo.
    
    Args:
        model_path: Caminho para o modelo salvo
        
    Returns:
        keras.Model: Modelo carregado ou None se erro
    """
    try:
        # Para carregar o modelo com custom objects
        custom_objects = {'OrthogonalRegularizer': OrthogonalRegularizer}
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Modelo carregado com sucesso de: {model_path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

# ======================================================================
# SALVAMENTO E CARREGAMENTO DO MODELO
# ======================================================================

# Salva o modelo treinado
save_model(model)

# Exemplo de como carregar o modelo (comentado)
# loaded_model = load_model("pointnet_model.keras")
# if loaded_model is not None:
#     # Teste do modelo carregado
#     test_custom_model(loaded_model, "monitor.obj", CLASS_MAP, NUM_POINTS)