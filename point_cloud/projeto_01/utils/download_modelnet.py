import os
import urllib.request
import zipfile

def download(dataset_url="http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip", data_dir="data"):
    """
    Baixa e extrai o dataset ModelNet10 para a pasta 'data' na raiz do projeto.
    """
    # Determina o caminho da raiz do projeto (a partir do diretório atual)
    # Para notebooks, geralmente é melhor usar o diretório atual como referência
    project_root = os.getcwd()
    while not os.path.exists(os.path.join(project_root, '.git')) and os.path.dirname(project_root) != project_root:
        project_root = os.path.dirname(project_root)
    
    # Caminho completo para o diretório de dados
    full_data_path = os.path.join(project_root, data_dir)
    
    # Cria a pasta data se não existir
    os.makedirs(full_data_path, exist_ok=True)
    
    # Nome do arquivo zip e caminho completo
    zip_filename = os.path.basename(dataset_url)
    zip_path = os.path.join(full_data_path, zip_filename)
    
    # Baixa o arquivo zip usando urllib
    print(f"Baixando {dataset_url} para {zip_path}...")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
    except Exception as e:
        print(f"Erro ao baixar o arquivo: {e}")
        return None
    
    # Extrai o arquivo zip
    print(f"Extraindo {zip_path} para {full_data_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(full_data_path)
        
        # Remove o arquivo zip após extração
        os.remove(zip_path)
        
        extracted_dir = os.path.join(full_data_path, "ModelNet10")
        print(f"Dataset ModelNet10 extraído para: {extracted_dir}")
        return extracted_dir
    except Exception as e:
        print(f"Erro ao extrair o arquivo: {e}")
        return None
