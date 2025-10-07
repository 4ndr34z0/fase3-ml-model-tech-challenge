import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union, Optional

# Importar lógica do modelo
import model_lgbm_logic as model_logic

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Variável global para armazenar os dados carregados
data_store: pd.DataFrame = None

# Pydantic Model para as linhas de dados (X) enviadas para a predição
# Não precisamos mais dos imports Optional e Union
from pydantic import BaseModel

# Pydantic Model para as linhas de dados (X) enviadas para a predição
class FeatureRow(BaseModel):
    # Todos os campos são obrigatórios (sem Optional) e com tipo único
    
    # SEQ_NFE: Veio como INT no JSON
    SEQ_NFE: int 
    
    # 1. FEATURES ORIGINAIS/COMPLETAS
    
    COD_UF_EMIT: str        # Veio como "PE" (str)
    TIP_FIN_NFE: int        # Veio como 1 (int)
    CEST_COMPLETO: int      # Veio como 0 (int)
    COD_CST: int            # Veio como 0 (int)
    NCM_COMPLETO: int       # Veio como 94034000 (int)
    CFOP_COMPLETO: int      # Veio como 6152 (int)
    EMIT_CNAE_COMPLETO: str # Veio como "AUSENTE" (str)
    DEST_CNAE_COMPLETO: str # Veio como "47539000" (str - Pydantic aceita str se o valor original for str)
    
    # Indicadores/Status
    EMIT_CRT: int           # Veio como 3 (int)
    EMIT_IND_SN: str        # Veio como "N" (str)
    DEST_SITUACAO: int      # Veio como 2 (int)
    DEST_IND_SN: int        # Veio como 0 (int)
    DEST_POSSUI_IE: int     # Veio como 1 (int)
    
    # 2. FEATURES DE DECOMPOSIÇÃO
    
    # CNAE Decomposição (Veio como "AUSENTE" (str) ou número puro (int/str))
    EMIT_CNAE_DIVISAO: str  
    EMIT_CNAE_GRUPO: str    
    EMIT_CNAE_CLASSE: str   
    DEST_CNAE_DIVISAO: str  # Veio como "47" (str)
    DEST_CNAE_GRUPO: str    # Veio como "475" (str)
    DEST_CNAE_CLASSE: str   # Veio como "4753" (str)
    
    # NCM Decomposição (Veio como INT)
    NCM_CAPITULO: int       
    NCM_POSICAO: int        
    NCM_SUBPOSICAO: int     
    
    # CFOP Decomposição (Veio como INT)
    CFOP_NATUREZA: int      
    CFOP_OPERACAO: int      
    
    # CEST Decomposição (Veio como INT ou STR)
    POSSUI_CEST: int        # Veio como 0 (int)
    CEST_SEGMENTO: str      # Veio como "00" (str)
    CEST_ITEM: str          # Veio como "00000" (str)

    # Chave interna do Frontend
    key: str
    
class PredictionRequest(BaseModel):
    data: List[FeatureRow]


@app.on_event("startup")
async def startup_event():
    """Carrega os dados da planilha no startup do servidor."""
    global data_store
    print("Iniciando o carregamento dos dados...")
    data_store = model_logic.load_data()
    print(f"Total de linhas carregadas: {len(data_store)}.")


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Tela inicial de pesquisa."""
    return templates.TemplateResponse("search_form.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search_nfe(request: Request, seq_nfe: str = Form(...)):
    """Busca o SEQ_NFE no DataFrame e exibe a tela de edição."""
    
    debug_line = 0

    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    if data_store is None or data_store.empty:
        return templates.TemplateResponse(
            "error_page.html", 
            {"request": request, "message": "Dados não carregados. Verifique o console do servidor."}, 
            status_code=500
        )
        
    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    try:
        seq_nfe_int = int(seq_nfe.strip())
    except ValueError:
        return templates.TemplateResponse(
            "search_form.html", 
            {"request": request, "error": f"O valor '{seq_nfe}' não é um número válido para SEQ_NFE."}, 
            status_code=400
        )

    results_df = data_store[data_store['SEQ_NFE'] == seq_nfe_int].copy()
    
    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    if results_df.empty:
        return templates.TemplateResponse(
            "search_form.html", 
            {"request": request, "error": f"Nenhuma nota encontrada com SEQ_NFE = {seq_nfe_int}."}, 
            status_code=404
        )

    debug_line += 1
    print(f"[DEBUG {debug_line}] Iniciando busca por SEQ_NFE = {seq_nfe}")
    # Prepara os dados para o template
    results_df['Tipo Imposto'] = "" # Adiciona a coluna vazia para o resultado
    display_columns = ['SEQ_NFE'] + model_logic.FEATURE_NAMES + ['Tipo Imposto']
    data_for_template = results_df[display_columns].to_dict('records')
    
    # print(data_for_template)    
    return templates.TemplateResponse(
        "editable_features.html", 
        {
            "request": request, 
            "data": data_for_template, 
            "features": model_logic.FEATURE_NAMES,
            "seq_nfe": seq_nfe
        }
    )


@app.post("/predict")
async def predict_rows(prediction_request: PredictionRequest):
    """Recebe os dados editados, executa o modelo e retorna o JSON com as predições."""
    
    if not prediction_request.data:
        return {"error": "Nenhum dado enviado para predição."}

    # 1. Converter Pydantic models para DataFrame
    data_list = [row.dict(exclude={'key', 'SEQ_NFE'}) for row in prediction_request.data]
    prediction_df = pd.DataFrame(data_list)

    # 2. Executar a predição
    classificacao_y = model_logic.predict_model(prediction_df)
    
    # 3. Mapear predições de volta para o identificador (key) do frontend
    results = {}
    for i, row_data in enumerate(prediction_request.data):
        # A chave 'key' é o identificador único criado no frontend
        row_key = row_data.key 
        results[row_key] = classificacao_y[i]
        
    return {"predictions": results} # Retorna JSON para o JavaScript