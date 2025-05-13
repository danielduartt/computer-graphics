import os
import tensorflow as tf

def download(dataset_url="http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip", data_dir="data"):
    """
    Baixa e extrai o dataset ModelNet10 para a pasta 'data' dentro do projeto.
    """
    # Cria a pasta 'data' se não existir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Faz o download e extrai o arquivo zip na pasta 'data'
    DATA_DIR = tf.keras.utils.get_file(
        fname="modelnet.zip",
        origin=dataset_url,
        extract=True,
        cache_dir=data_dir  # Define a pasta onde os arquivos extraídos serão armazenados
    )
    
    # Define o diretório onde o ModelNet10 foi extraído
    extracted_dir = os.path.join(os.path.dirname(DATA_DIR), "datasets", "ModelNet10")
    
    print(f"Dataset ModelNet10 extraído para: {extracted_dir}")
    return extracted_dir