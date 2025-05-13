import subprocess

def update_requirements(output_file="requirements.txt"):
    """
    Atualiza o arquivo requirements.txt com os pacotes atualmente instalados.
    """
    try:
        with open(output_file, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        print(f"✅ Requisitos atualizados em '{output_file}'")
    except Exception as e:
        print(f"❌ Erro ao atualizar requirements.txt: {e}")
