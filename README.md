# Projeto de Classificação de Notas Fiscais (FastAPI + LightGBM)

Este projeto implementa uma interface web simples usando **FastAPI** para buscar dados de notas fiscais por `SEQ_NFE`, permitir a edição de *features* e classificar as linhas usando um modelo de Machine Learning **LightGBM** persistido.

## Estrutura do Projeto

* `main.py`: Servidor principal e endpoints de API.
* `model_lgbm_logic.py`: Lógica de carregamento de dados (`.csv`) e execução do modelo (`.txt`).
* `templates/`: Arquivos HTML (Jinja2) para a interface do usuário.
* `modelo_lgbm 1.txt`: Modelo LightGBM treinado.
* `requirements.txt`: Dependências Python para instalação em ambiente de produção (Render).
* `render.yaml`: Definição da infraestrutura para deploy no Render.

## Variáveis do Modelo (Features)

O modelo utiliza 12 features para classificação:
`COD_UF_EMIT`, `TIP_FIN_NFE`, `COD_CEST`, `COD_CST`, `COD_NCM`, `COD_CFOP`, `EMIT_COD_CNAE`, `EMIT_CRT`, `EMIT_IND_SN`, `DEST_CNAE_PRINC`, `DEST_POSSUI_IE`, `DEST_SIMPLES`.

## Como Executar Localmente

1. **Instalar dependências:**

```bash
pip install -r requirements.txt
```

2. **Executar o uniconr:**
`uvicorn main:app --reload`

Acessar: `http://127.0.0.1:8000/`

## Como Executar no Render

1. Instalar as dependências de requirements.txt.

2. Executar o comando uvicorn 
`main:app --host 0.0.0.0 --port $PORT`