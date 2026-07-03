# generate_pipelines_ontologia_real.py
"""
Generador de pipelines usando AGENTES REALES de GraphDB
El LLM elige los agentes y justifica su elección
"""

import json
import shutil
import re
from pathlib import Path
import requests
import sys

sys.path.insert(0, str(Path(__file__).parent))
from main3 import fetch_agent_fragments

# ============================================================
# CONFIGURACIÓN
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:latest"

DATA_DIR = Path(__file__).parent / "data" / "uploads"
KAGGLE_BASE = Path(__file__).parent / "kaggle_datasets-main" / "datasets"

# Servidores MCP válidos
VALID_SERVERS = {
    "data_loading", "data_preparation", "model_training",
    "model_evaluation", "feature_engineering", "mathematics"
}

# MODELOS VÁLIDOS EN TU SERVIDOR
CLASSIFICATION_MODELS = [
    "logistic_regression", "decision_tree", "random_forest", 
    "svm", "knn", "gradient_boosting", "naive_bayes", "xgboost"
]

REGRESSION_MODELS = [
    "linear", "ridge", "lasso", "decision_tree", 
    "random_forest", "svr", "knn", "gradient_boosting"
]

# ============================================================
# OBTENER AGENTES REALES DE GRAPHDB
# ============================================================
def get_ontology_agents():
    fragments = fetch_agent_fragments()
    if not fragments:
        return []
    
    agents = []
    for f in fragments:
        if isinstance(f, str):
            match = re.search(r'Nombre:\s*([^;]+)', f)
            if match:
                agents.append(match.group(1).strip())
    return agents

def get_agents_for_llm():
    agents = get_ontology_agents()
    if not agents:
        return "No se encontraron agentes en GraphDB"
    
    result = "AGENTES ONTOLÓGICOS DISPONIBLES:\n"
    for agent in agents:
        result += f"- {agent}\n"
    return result

def get_mcp_context():
    return f"""SERVIDORES MCP DISPONIBLES:

1. data_loading - Herramientas: load_csv, validate_csv_structure, detect_missing_values, get_basic_stats

2. data_preparation - Herramientas: handle_missing_values, remove_duplicates, scale_features, encode_categorical, split_dataset

3. model_training - Herramientas: train_classification_model, train_regression_model
   - MODELOS VÁLIDOS para clasificación: {', '.join(CLASSIFICATION_MODELS)}
   - MODELOS VÁLIDOS para regresión: {', '.join(REGRESSION_MODELS)}

4. model_evaluation - Herramientas: evaluate_classification_model, evaluate_regression_model

5. feature_engineering - Herramientas: feature_selection

6. mathematics - Herramientas: analyze_csv, mean, median, standard_deviation
"""

# ============================================================
# DATASETS
# ============================================================
CUSTOM_DATASETS = [
    # ========== CLASSIFICATION ==========
    # Easy
    {
        "name": "EEG_Eye_State_Classification.csv",
        "task": "classification",
        "difficulty": "easy",
        "target": "eyeDetection",
        "query": "clasificación de datos médicos"
    },
    {
        "name": "heart.csv",
        "task": "classification",
        "difficulty": "easy",
        "target": "target",
        "query": "clasificación de enfermedades cardíacas"
    },
    # Mid
    {
        "name": "personality_synthetic_dataset.csv",
        "task": "classification",
        "difficulty": "mid",
        "target": "stress_handling",
        "query": "clasificación de nivel de estrés"
    },
    # Hard
    {
        "name": "emails.csv",
        "task": "classification",
        "difficulty": "hard",
        "target": "Prediction",
        "query": "detección de spam en emails"
    },
    {
        "name": "Music_recommendation_withProperRagas.csv",
        "task": "classification",
        "difficulty": "hard",
        "target": "instrument",
        "query": "clasificación de instrumento usado"
    },
    # ========== REGRESSION ==========
    # Easy
    {
        "name": "boston.csv",
        "task": "regression",
        "difficulty": "easy",
        "target": "MEDV",
        "query": "predicción de precios de viviendas"
    },
    {
        "name": "winequality.csv",
        "task": "regression",
        "difficulty": "easy",
        "target": "quality",
        "query": "predicción de calidad del vino"
    },
    # Mid
    {
        "name": "diamonds.csv",
        "task": "regression",
        "difficulty": "mid",
        "target": "total_sales_price",
        "query": "predicción de precio de venta de diamantes"
    },
    {
        "name": "student_performance.csv",
        "task": "regression",
        "difficulty": "mid",
        "target": "grade",
        "query": "predicción de calificaciones de estudiantes"
    },
    # Hard
    {
        "name": "starbucks_customer_ordering_patterns.csv",
        "task": "regression",
        "difficulty": "hard",
        "target": "customer_satisfaction",
        "query": "predicción de satisfacción del cliente (1-5)"
    },
]

# ============================================================
# PROMPT - INSTRUCCIÓN ESTRICTA DE UN SOLO JSON
# ============================================================
def build_prompt(dataset, exec_id, agents_context, mcp_context):
    name = dataset["name"]
    task = dataset["task"]
    target = dataset["target"]
    difficulty = dataset["difficulty"]
    query = dataset["query"]
    model_path = f"models/{name.replace('.csv','')}_exec{exec_id}.pkl"
    
    # Lista de modelos según tarea
    if task == "classification":
        valid_models = CLASSIFICATION_MODELS
        default_model = "random_forest"
    else:
        valid_models = REGRESSION_MODELS
        default_model = "linear"
    
    return f"""Eres un orquestador experto de pipelines.

{agents_context}

{mcp_context}

TAREA: {task} en {name}
TARGET: {target}
DIFICULTAD: {difficulty}
DESCRIPCIÓN: {query}

INSTRUCCIONES ESTRICTAS:
- RESPUESTA: SOLO UN JSON, nada más
- Número de pasos: TÚ decides (3-8 pasos)
- Agentes: SOLO de la lista AGENTES ONTOLÓGICOS
- model_type válidos para {task}: {valid_models}

EJEMPLO DE JSON VÁLIDO:
{{"analisis_general": "texto", "pipeline": [{{"paso": 1, "agente_ontologico": "data_loader", "server": "data_loading", "tool_name": "load_csv", "arguments": {{"filepath": "{name}"}}, "explicacion": "texto"}}]}}

Genera el JSON ahora (SOLO JSON, sin texto antes o después):"""

# ============================================================
# FUNCIÓN MEJORADA PARA EXTRAER JSON
# ============================================================
def extract_json(text):
    """Extrae el primer JSON válido de un texto"""
    # Buscar el primer { y hacer matching de braces
    start = text.find('{')
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
        
        if char == '\\':
            escape = True
            continue
        
        if char == '"' and not escape:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
    
    return None

def clean_and_parse_json(text):
    """Limpia y parsea JSON de manera robusta"""
    # Intentar extraer el primer JSON válido
    json_str = extract_json(text)
    
    if not json_str:
        # Buscar con regex como fallback
        matches = re.findall(r'\{[^{}]*\}(?:\{[^{}]*\})*', text, re.DOTALL)
        if matches:
            json_str = matches[0]
    
    if not json_str:
        return None
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Intentar limpiar caracteres problemáticos
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        try:
            return json.loads(cleaned)
        except:
            return None

# ============================================================
# FUNCIONES DE CORRECCIÓN
# ============================================================
def fix_model_type(model_type, task):
    if not model_type:
        return "random_forest" if task == "classification" else "linear"
    
    model_lower = model_type.lower().strip()
    
    classification_map = {
        "random forest": "random_forest", "random forest classifier": "random_forest",
        "randomforest": "random_forest", "rf": "random_forest",
        "logistic regression": "logistic_regression", "logisticregression": "logistic_regression",
        "decision tree": "decision_tree", "decision_tree_classifier": "decision_tree",
        "svm": "svm", "svm classifier": "svm",
        "knn": "knn", "k-nearest neighbors": "knn",
        "gradient boosting": "gradient_boosting", "gbm": "gradient_boosting",
        "xgboost": "xgboost", "xgb": "xgboost",
        "naive bayes": "naive_bayes"
    }
    
    regression_map = {
        "linear": "linear", "linear regression": "linear",
        "ridge": "ridge", "ridge regression": "ridge",
        "lasso": "lasso", "lasso regression": "lasso",
        "random forest": "random_forest", "random forest regressor": "random_forest",
        "decision tree": "decision_tree", "svr": "svr",
        "knn": "knn", "gradient boosting": "gradient_boosting"
    }
    
    map_to_use = classification_map if task == "classification" else regression_map
    
    for wrong, correct in map_to_use.items():
        if wrong in model_lower:
            return correct
    
    return "random_forest" if task == "classification" else "linear"

def fix_step(step, dataset, exec_id):
    name = dataset["name"]
    target = dataset["target"]
    task = dataset["task"]
    model_path = f"models/{name.replace('.csv','')}_exec{exec_id}.pkl"
    
    if "arguments" not in step:
        step["arguments"] = {}
    
    args = step["arguments"]
    tool = step.get("tool_name", "")
    
    if "model_type" in args:
        args["model_type"] = fix_model_type(args["model_type"], task)
    
    if tool in ["load_csv", "validate_csv_structure", "detect_missing_values", "get_basic_stats"]:
        if "filepath" not in args:
            args["filepath"] = name
    
    if tool in ["train_classification_model", "train_regression_model"]:
        if "target_column" not in args:
            args["target_column"] = target
        if "model_save_path" not in args:
            args["model_save_path"] = model_path
        if "filepath" not in args:
            args["filepath"] = name
        if "test_size" not in args:
            args["test_size"] = 0.2
        if "random_state" not in args:
            args["random_state"] = 42
        if "model_type" not in args:
            args["model_type"] = "random_forest" if task == "classification" else "linear"
    
    if tool in ["evaluate_classification_model", "evaluate_regression_model"]:
        if "model_path" not in args:
            args["model_path"] = model_path
        if "test_data_path" not in args:
            args["test_data_path"] = name
    
    step["arguments"] = args
    return step

# ============================================================
# GENERAR PIPELINE
# ============================================================
def generate_pipeline(dataset, exec_id, agents_context, mcp_context):
    prompt = build_prompt(dataset, exec_id, agents_context, mcp_context)
    
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.4,
            "num_predict": 4096
        }, timeout=180)
        
        if resp.status_code != 200:
            print(f" HTTP {resp.status_code}")
            return None
        
        response_text = resp.json().get("response", "")
        
        # Usar la función robusta de extracción
        pipeline = clean_and_parse_json(response_text)
        
        if not pipeline:
            print(f" No se pudo extraer JSON")
            return None
        
        if "pipeline" not in pipeline:
            print(f" JSON sin pipeline")
            return None
        
        # Corregir pasos
        for i, step in enumerate(pipeline.get("pipeline", [])):
            step["paso"] = i + 1
            step = fix_step(step, dataset, exec_id)
        
        return pipeline
        
    except Exception as e:
        print(f" Error: {e}")
        return None

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("🚀 GENERADOR DE PIPELINES CON ONTOLOGÍA")
    print("="*60)
    
    agents_list = get_ontology_agents()
    agents_context = get_agents_for_llm()
    mcp_context = get_mcp_context()
    
    print(f"\n✅ AGENTES ONTOLÓGICOS: {len(agents_list)}")
    print(f"   Ejemplo: {', '.join(agents_list[:5])}...")
    
    print(f"\n✅ SERVIDORES MCP: {len(VALID_SERVERS)}")
    print(f"✅ MODELOS CLASIFICACIÓN: {CLASSIFICATION_MODELS}")
    print(f"✅ MODELOS REGRESIÓN: {REGRESSION_MODELS}")
    
    # Copiar datasets
    for ds in CUSTOM_DATASETS:
        dest = DATA_DIR / ds["name"]
        if not dest.exists():
            for task_type in ["classification", "regression"]:
                for diff in ["easy", "mid", "hard"]:
                    src = KAGGLE_BASE / task_type / diff / ds["name"]
                    if src.exists():
                        shutil.copy2(src, dest)
                        print(f"   📁 Copiado: {ds['name']}")
                        break
    
    print(f"\n📊 DATASETS: {len(CUSTOM_DATASETS)}")
    
    pipelines = []
    exec_id = 0
    
    for ds in CUSTOM_DATASETS:
        print(f"\n{'='*50}")
        print(f"📁 {ds['name']} ({ds['task']}) - Target: {ds['target']}")
        
        for i in range(2):
            print(f"   Pipeline {i+1}/2 generando...", end="", flush=True)
            
            pipeline = generate_pipeline(ds, exec_id, agents_context, mcp_context)
            
            if pipeline is None:
                print(" ❌ FALLÓ")
                continue
            
            pipelines.append({
                "filename": ds["name"],
                "task_type": ds["task"],
                "difficulty": ds["difficulty"],
                "target_column": ds["target"],
                "query_used": ds["query"],
                "execution_id": exec_id,
                "pipeline": pipeline
            })
            exec_id += 1
            print(f" ✅ {len(pipeline.get('pipeline', []))} pasos")
    
    if pipelines:
        out = Path(__file__).parent / "pipelines_mcp.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pipelines, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✅ GENERADOS {len(pipelines)} PIPELINES")
        print(f"📁 Guardados en: {out}")
    else:
        print("\n❌ No se generó ningún pipeline")

if __name__ == "__main__":
    main()