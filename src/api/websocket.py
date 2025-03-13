import json
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .models import TaskRequest
from .agent_manager import AgentManager

app = FastAPI()

# Store active connections
connections: Dict[str, AgentManager] = {}

@app.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(id(websocket))
    agent_manager = AgentManager(websocket)
    connections[connection_id] = agent_manager

    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            try:
                # Parse the request
                request = TaskRequest.model_validate_json(message)

                # Initialize and run the agent
                await agent_manager.initialize_agent(
                    task=request.task,
                    add_infos=request.add_infos,
                    config=request.config
                )
                await agent_manager.run_task()

            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "details": "The message must be a valid JSON object"
                })
            except Exception as e:
                await websocket.send_json({
                    "error": "Task execution failed",
                    "details": str(e)
                })

    except WebSocketDisconnect:
        # Clean up resources when client disconnects
        if connection_id in connections:
            await agent_manager.cleanup()
            del connections[connection_id]

@app.websocket("/agent/stop/{connection_id}")
async def stop_agent(websocket: WebSocket, connection_id: str):
    await websocket.accept()
    try:
        if connection_id in connections:
            agent_manager = connections[connection_id]
            await agent_manager.stop_task()
            await websocket.send_json({"status": "success", "message": "Stop request sent"})
        else:
            await websocket.send_json({"error": "Connection not found"})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close() 