# execute_pipelines.py
import asyncio
import json
import httpx
from pathlib import Path

MCP_SERVERS = {
    "mathemathics": "http://127.0.0.1:8001/mcp",
    "data_loading": "http://127.0.0.1:8002/mcp",
    "data_preparation": "http://127.0.0.1:8003/mcp",
    "model_training": "http://127.0.0.1:8004/mcp",
    "model_evaluation": "http://127.0.0.1:8005/mcp",
    "feature_engineering": "http://127.0.0.1:8006/mcp",
}

async def call_tool(server_url: str, tool_name: str, arguments: dict, timeout=300) -> dict:
    """Llama a una herramienta MCP correctamente (con inicialización de sesión)"""
    
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # 1. Inicializar conexión
            init = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {},
                    "clientInfo": {"name": "runner", "version": "1.0"}
                },
                "id": 1
            }
            
            resp = await client.post(server_url, json=init, headers=headers)
            session_id = resp.headers.get("mcp-session-id")
            
            if not session_id:
                return {"error": "No se pudo obtener session_id"}
            
            # 2. Llamar a la herramienta con la sesión
            headers_with_session = headers.copy()
            headers_with_session["mcp-session-id"] = session_id
            
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 2
            }
            
            resp = await client.post(server_url, json=payload, headers=headers_with_session)
            
            # 3. Parsear respuesta SSE
            for line in resp.text.split('\n'):
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if "result" in data:
                        for item in data["result"].get("content", []):
                            if item.get("type") == "text":
                                try:
                                    return json.loads(item["text"])
                                except:
                                    return {"raw": item["text"]}
                    if "error" in data:
                        return {"error": data["error"].get("message", "Unknown error")}
            
            return {"error": "No se pudo parsear respuesta"}
            
        except httpx.TimeoutException:
            return {"error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

async def execute_pipeline(pipeline_info):
    steps = pipeline_info.get("pipeline", {}).get("pipeline", [])
    filename = pipeline_info.get("filename", "unknown")
    task_type = pipeline_info.get("task_type", "classification")
    target = pipeline_info.get("target_column", "target")
    
    print(f"\n{'='*60}")
    print(f"📊 {filename} ({task_type})")
    print(f"{'='*60}")
    
    results = {"filename": filename, "task_type": task_type, "target": target, "steps": [], "metrics": {}}
    
    for step in steps:
        server = step.get("server")
        tool = step.get("tool_name")
        args = step.get("arguments", {}).copy()
        
        # Asegurar argumentos obligatorios
        if tool in ["load_csv", "validate_csv_structure", "detect_missing_values"]:
            if "filepath" not in args:
                args["filepath"] = filename
        
        if tool == "split_dataset":
            if "filepath" not in args:
                args["filepath"] = filename
            if "train_ratio" not in args:
                args["train_ratio"] = 0.8
        
        if tool in ["train_classification_model", "train_regression_model"]:
            if "filepath" not in args:
                args["filepath"] = filename
            if "target_column" not in args:
                args["target_column"] = target
            if "model_type" not in args:
                args["model_type"] = "random_forest"
            if "test_size" not in args:
                args["test_size"] = 0.2
            if "random_state" not in args:
                args["random_state"] = 42
        
        if tool in ["evaluate_classification_model", "evaluate_regression_model"]:
            if "model_path" not in args:
                args["model_path"] = f"models/{filename.replace('.csv','')}_model.pkl"
            if "test_data_path" not in args:
                args["test_data_path"] = filename
            if "target_column" not in args:
                args["target_column"] = target
        
        print(f"  📌 {tool}")
        
        if server not in MCP_SERVERS:
            results["metrics"]["error"] = f"Servidor {server} no existe"
            break
        
        result = await call_tool(MCP_SERVERS[server], tool, args)
        
        if "error" in result:
            print(f"     ❌ {result['error']}")
            results["metrics"]["error"] = result["error"]
            break
        else:
            print(f"     ✅ Completado")
            
            if tool == "evaluate_classification_model":
                results["metrics"] = {
                    "accuracy": result.get("accuracy"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1_score": result.get("f1_score"),
                    "confusion_matrix": result.get("confusion_matrix"),
                    "test_samples": result.get("test_samples"),
                    "n_classes": result.get("n_classes"),
                    "is_binary": result.get("is_binary")
                }
                print(f"     📈 accuracy={results['metrics']['accuracy']}, f1={results['metrics']['f1_score']}")
            elif tool == "evaluate_regression_model":
                results["metrics"] = {
                    "r2_score": result.get("r2_score"),
                    "mse": result.get("mse"),
                    "rmse": result.get("rmse"),
                    "mae": result.get("mae"),
                    "mape": result.get("mape")
                }
                print(f"     📈 r2={results['metrics']['r2_score']}, rmse={results['metrics']['rmse']}")
            elif tool in ["train_classification_model", "train_regression_model"]:
                if "train_accuracy" in result:
                    print(f"     📊 train_acc={result.get('train_accuracy')}, test_acc={result.get('test_accuracy')}")
                if "train_r2" in result:
                    print(f"     📊 train_r2={result.get('train_r2')}, test_r2={result.get('test_r2')}")
    
    return results

async def main():
    print("="*60)
    print("🚀 EJECUTOR DE PIPELINES MCP")
    print("="*60)
    
    # Verificar servidores
    print("\n🔍 Verificando servidores...")
    for name, url in MCP_SERVERS.items():
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(url)
                print(f"   ✅ {name}" if resp.status_code in [200, 406] else f"   ⚠️ {name}")
        except:
            print(f"   ❌ {name} no responde")
    
    pipeline_file = Path(__file__).parent / "pipelines_mcp.json"
    
    if not pipeline_file.exists():
        print(f"\n❌ No se encuentra {pipeline_file}")
        print("Ejecuta primero: python generate_pipelines_finalv4.py")
        return
    
    with open(pipeline_file, "r", encoding="utf-8") as f:
        pipelines = json.load(f)
    
    print(f"\n📁 {len(pipelines)} pipelines generados")
    
    # Ejecutar primeros 3 como prueba
    results = []
    for i, p in enumerate(pipelines[:36]):
        print(f"\n--- Pipeline {i+1}/3 ---")
        res = await execute_pipeline(p)
        results.append(res)
    
    out_file = Path(__file__).parent / "resultados_ejecucion_mcpv2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✅ Resultados guardados en {out_file}")

if __name__ == "__main__":
    asyncio.run(main())