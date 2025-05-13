# 🎓 Computação Gráfica – Projetos Acadêmicos

Este repositório reúne os **trabalhos práticos** e os **materiais teóricos** desenvolvidos na disciplina de **Computação Gráfica** com o professor dr.  HAROLDO GOMES BARROSO FILHO do curso de Engenharia da Computação - UFMA . Cada diretório corresponde a um tema abordado, com códigos, visualizações e documentação associados.

---

## 📁 Estrutura

- `point_cloud/` – Geração e visualização de nuvem de pontos
- `voxelization/` – Voxelização de malhas 3D
- `shaders/` – Sombreamento e renderização básica
- `surface_reconstruction/` – Reconstrução de superfícies a partir de pontos
- `mesh_to_cloud/` – Conversão de malhas para nuvem de pontos
- `docs/` – Materiais teóricos explicando cada técnica
- `utils/` – Scripts auxiliares e loaders
- `assets/` – Figuras e resultados para visualização
- `data/` – Dados utilizados nos projetos (não incluídos no repositório)

---

## 🚀 Como começar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
````

### 2. Baixe o dataset (ModelNet10)

O conjunto de dados completo **não é incluído neste repositório** devido ao seu tamanho.

Você pode baixá-lo automaticamente com o script:

```bash
python utils/download_modelnet.py
```

Os dados extraídos devem ser colocados dentro da pasta `data/ModelNet10/`.

---

## 🧪 Projetos incluídos

| Pasta                     | Tema                        | Descrição                                |
| ------------------------- | --------------------------- | ---------------------------------------- |
| `point_cloud/`            | Nuvem de Pontos             | Amostragem e visualização de malhas      |
| `voxelization/`           | Voxelização                 | Representação volumétrica com voxels     |
| `shaders/`                | Shaders                     | Sombreamento e efeitos gráficos básicos  |
| `surface_reconstruction/` | Reconstrução de Superfícies | Reconstrução usando Poisson ou similares |
| `mesh_to_cloud/`          | Malha para Nuvem            | Conversão por vértices ou faces          |

---

## 📚 Documentação Teórica

Os conteúdos teóricos que embasam cada trabalho estão na pasta [`docs/`](docs/), e incluem:

* Explicações dos métodos utilizados
* Equações e fundamentos

---