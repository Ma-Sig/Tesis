# run_all_mcp_servers.py
"""
Script para ejecutar los 6 servidores MCP simultáneamente.
Todos los servidores deben estar en la carpeta 'server' con sus respectivos nombres.
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path
import threading

# Configuración de los servidores MCP
MCP_SERVERS = [
    {
        "name": "data_loading",
        "script": "server.py",
        "port": 8002,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[94m",  # Azul
    },
    {
        "name": "mathematics",
        "script": "server.py",
        "port": 8001,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[92m",  # Verde
    },
    {
        "name": "data_preparation",
        "script": "server.py",
        "port": 8003,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[93m",  # Amarillo
    },
    {
        "name": "model_training",
        "script": "server.py",
        "port": 8004,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[95m",  # Magenta
    },
    {
        "name": "model_evaluation",
        "script": "server.py",
        "port": 8005,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[96m",  # Cyan
    },
    {
        "name": "feature_engineering",
        "script": "server.py",
        "port": 8006,
        "transport": "http",
        "host": "127.0.0.1",
        "color": "\033[91m",  # Rojo
    },
]

# Colores para la terminal
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"

# Ruta base donde están las carpetas de los servidores
BASE_PATH = Path(__file__).parent / "server"

class MCPServerManager:
    """Gestiona la ejecución de múltiples servidores MCP"""
    
    def __init__(self, servers_config):
        self.servers_config = servers_config
        self.processes = {}
        self.running = False
    
    def find_server_script(self, server_name):
        """Busca el script server.py en la carpeta del servidor"""
        server_folder = BASE_PATH / server_name
        script_path = server_folder / "server.py"
        
        if script_path.exists():
            return script_path
        else:
            # Buscar en subcarpetas
            for py_file in server_folder.rglob("server.py"):
                return py_file
        return None
    
    def start_server(self, server_config):
        """Inicia un servidor MCP"""
        name = server_config["name"]
        port = server_config["port"]
        transport = server_config["transport"]
        host = server_config["host"]
        color = server_config.get("color", "\033[0m")
        
        # Encontrar el script
        script_path = self.find_server_script(name)
        
        if not script_path:
            print(f"{RED}❌ Error: No se encontró server.py para '{name}' en {BASE_PATH / name}{RESET}")
            return None
        
        # Construir comando
        cmd = [
            sys.executable,
            str(script_path),
            "--transport", transport,
            "--host", host,
            "--port", str(port)
        ]
        
        try:
            # Iniciar proceso
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=script_path.parent
            )
            
            print(f"{color}🚀 Iniciando {name} en puerto {port}...{RESET}")
            
            return process
            
        except Exception as e:
            print(f"{RED}❌ Error iniciando {name}: {e}{RESET}")
            return None
    
    def start_all(self):
        """Inicia todos los servidores"""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}🚀 INICIANDO SERVIDORES MCP{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
        
        for server in self.servers_config:
            process = self.start_server(server)
            if process:
                self.processes[server["name"]] = {
                    "process": process,
                    "config": server,
                    "port": server["port"]
                }
            time.sleep(2)  # Esperar entre servidores
        
        self.running = True
        self.show_status()
    
    def stop_server(self, name):
        """Detiene un servidor específico"""
        if name in self.processes:
            process = self.processes[name]["process"]
            color = self.processes[name]["config"].get("color", "\033[0m")
            
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"{color}🛑 Servidor {name} detenido{RESET}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"{color}⚠️ Servidor {name} forzado a detenerse{RESET}")
            
            del self.processes[name]
    
    def stop_all(self):
        """Detiene todos los servidores"""
        print(f"\n{BOLD}{YELLOW}🛑 Deteniendo todos los servidores...{RESET}")
        
        for name in list(self.processes.keys()):
            self.stop_server(name)
        
        self.running = False
        print(f"{GREEN}✅ Todos los servidores detenidos{RESET}")
    
    def show_status(self):
        """Muestra el estado de los servidores"""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}📊 ESTADO DE SERVIDORES{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        
        for name, info in self.processes.items():
            config = info["config"]
            color = config.get("color", "\033[0m")
            port = config["port"]
            process = info["process"]
            
            status = f"{GREEN}● RUNNING{RESET}" if process.poll() is None else f"{RED}● STOPPED{RESET}"
            print(f"{color}  📦 {name:<20} {RESET} Puerto: {port:<5} {status}")
        
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    def monitor_output(self):
        """Monitorea la salida de todos los servidores en hilos separados"""
        def monitor(server_name, process, color):
            for line in process.stdout:
                if line.strip():
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"{color}[{timestamp}] [{server_name}]{RESET} {line.strip()}")
        
        monitors = []
        for name, info in self.processes.items():
            process = info["process"]
            color = info["config"].get("color", "\033[0m")
            
            monitor_thread = threading.Thread(
                target=monitor,
                args=(name, process, color),
                daemon=True
            )
            monitor_thread.start()
            monitors.append(monitor_thread)
        
        return monitors
    
    def wait_for_servers(self, timeout=30):
        """Espera a que todos los servidores estén listos"""
        import socket
        
        print(f"\n{BOLD}{BLUE}⏳ Esperando que los servidores estén listos...{RESET}")
        
        for name, info in self.processes.items():
            port = info["port"]
            ready = False
            start_time = time.time()
            
            while not ready and (time.time() - start_time) < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    sock.close()
                    
                    if result == 0:
                        print(f"{GREEN}✅ {name} listo en puerto {port}{RESET}")
                        ready = True
                    else:
                        time.sleep(0.5)
                except:
                    time.sleep(0.5)
            
            if not ready:
                print(f"{YELLOW}⚠️ {name} no respondió en {timeout}s{RESET}")


def print_banner():
    """Imprime el banner de inicio"""
    banner = f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════╗
║                    🚀 MCP SERVERS MANAGER 🚀                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Servidores disponibles:                                          ║
║    📦 data_loading      → Puerto 8002  (Carga de datos)          ║
║    📦 mathematics       → Puerto 8001  (Operaciones matemáticas) ║
║    📦 data_preparation  → Puerto 8003  (Preparación de datos)    ║
║    📦 model_training    → Puerto 8004  (Entrenamiento de modelos)║
║    📦 model_evaluation  → Puerto 8005  (Evaluación de modelos)   ║
║    📦 feature_engineering→ Puerto 8006 (Ingeniería de features)  ║
╠══════════════════════════════════════════════════════════════════╣
║  Comandos:                                                        ║
║    • Presiona Ctrl+C para detener todos los servidores           ║
║    • Los logs se muestran en tiempo real                         ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def check_server_folders():
    """Verifica que existan las carpetas de los servidores"""
    missing_folders = []
    
    for server in MCP_SERVERS:
        folder = BASE_PATH / server["name"]
        if not folder.exists():
            missing_folders.append(server["name"])
    
    if missing_folders:
        print(f"{RED}❌ ERROR: Carpetas de servidores no encontradas:{RESET}")
        for folder in missing_folders:
            print(f"   - {BASE_PATH / folder}")
        print(f"\n{YELLOW}💡 Sugerencia: Asegúrate de que los servidores estén en:{RESET}")
        print(f"   {BASE_PATH}")
        return False
    
    return True


def main():
    """Función principal"""
    # Verificar carpetas
    if not check_server_folders():
        sys.exit(1)
    
    # Imprimir banner
    print_banner()
    
    # Crear manager
    manager = MCPServerManager(MCP_SERVERS)
    
    try:
        # Iniciar todos los servidores
        manager.start_all()
        
        # Iniciar monitoreo de salida
        monitors = manager.monitor_output()
        
        # Esperar a que los servidores estén listos
        manager.wait_for_servers(timeout=30)
        
        # Mostrar información de conexión
        print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
        print(f"{BOLD}{GREEN}✨ SERVIDORES MCP INICIADOS CORRECTAMENTE ✨{RESET}")
        print(f"{BOLD}{GREEN}{'='*60}{RESET}")
        print(f"\n{BOLD}📍 Endpoints disponibles:{RESET}")
        for server in MCP_SERVERS:
            print(f"   • http://{server['host']}:{server['port']}/mcp")
        
        print(f"\n{BOLD}{YELLOW}💡 Presiona Ctrl+C para detener todos los servidores{RESET}\n")
        
        # Mantener el script ejecutándose
        while manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{BOLD}{YELLOW}⚠️ Interrupción recibida{RESET}")
        manager.stop_all()
        print(f"{GREEN}✅ Servidores detenidos. ¡Hasta luego!{RESET}\n")
        sys.exit(0)
    except Exception as e:
        print(f"{RED}❌ Error inesperado: {e}{RESET}")
        manager.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()