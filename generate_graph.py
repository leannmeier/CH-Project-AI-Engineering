from src.graph import get_multi_agent_executor

def generate_architecture_diagram():
    print("Generando diagrama de arquitectura del orquestador...")
    
    # Obtenemos el grafo compilado
    app = get_multi_agent_executor()
    
    try:
        # Genera los bytes de la imagen PNG utilizando la representación Mermaid de LangGraph
        png_bytes = app.get_graph().draw_mermaid_png()
        
        output_path = "graph_architecture.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
            
        print(f"¡Éxito! Diagrama guardado correctamente como '{output_path}'.")
    except Exception as e:
        print(f"Error al generar el diagrama (asegúrate de tener conexión a internet): {e}")

if __name__ == "__main__":
    generate_architecture_diagram()