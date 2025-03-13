import json
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException
from .models import TaskRequest, MessageType
from .agent_state import state_manager
from src.utils.default_config_settings import default_config
from src.utils import utils
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

app = FastAPI()

# Store active connections
connections: Dict[str, WebSocket] = {}

@app.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        connection_id = str(id(websocket))
        
        # Register the connection
        ws_handler = await state_manager.register_connection(connection_id, websocket)
        
        try:
            while True:
                # Wait for messages
                message = await websocket.receive_text()
                
                try:
                    # Parse the request
                    request = TaskRequest.model_validate_json(message)
                    
                    # Send initial status
                    await ws_handler.send_update(
                        MessageType.STATUS,
                        {"status": "connected", "message": "WebSocket connection established"}
                    )
                    
                    await ws_handler.send_update(
                        MessageType.STATUS,
                        {"status": "initializing", "message": "Initializing agent..."}
                    )

                    # Clear any previous stop request
                    state_manager.clear_stop()

                    # Initialize browser if needed
                    browser_config = BrowserConfig(
                        headless=request.config.get('headless', True),
                        disable_security=request.config.get('disable_security', False),
                        extra_chromium_args=[f"--window-size={request.config.get('window_w', 1280)},{request.config.get('window_h', 720)}"]
                    )
                    await state_manager.initialize_browser(browser_config)

                    # Initialize context if needed
                    context_config = BrowserContextConfig(
                        trace_path=request.config.get('save_trace_path'),
                        save_recording_path=request.config.get('save_recording_path'),
                        no_viewport=False,
                        browser_window_size={
                            "width": request.config.get('window_w', 1280),
                            "height": request.config.get('window_h', 720)
                        }
                    )
                    await state_manager.initialize_context(context_config)

                    # Get LLM model
                    llm = utils.get_llm_model(
                        provider=request.config.get('llm_provider', 'openai'),
                        model_name=request.config.get('llm_model_name'),
                        num_ctx=request.config.get('llm_num_ctx'),
                        temperature=request.config.get('llm_temperature'),
                        base_url=request.config.get('llm_base_url'),
                        api_key=request.config.get('llm_api_key'),
                    )

                    # Initialize agent if needed
                    if state_manager.state.agent is None:
                        from src.agent.custom_agent import CustomAgent
                        state_manager.state.agent = CustomAgent(
                            task=request.task,
                            add_infos=request.add_infos or "",
                            llm=llm,
                            browser=state_manager.state.browser,
                            browser_context=state_manager.state.browser_context,
                            use_vision=request.config.get('use_vision', False),
                            max_actions_per_step=request.config.get('max_actions_per_step', 5),
                            tool_calling_method=request.config.get('tool_calling_method', 'auto')
                        )

                    await ws_handler.send_update(
                        MessageType.STATUS,
                        {"status": "running", "message": "Starting task execution"}
                    )

                    # Run the agent
                    history = await state_manager.state.agent.run(
                        max_steps=request.config.get('max_steps', 100)
                    )

                    # Send results
                    await ws_handler.send_update(
                        MessageType.RESULT,
                        {
                            "final_result": history.final_result(),
                            "errors": history.errors(),
                            "model_actions": history.model_actions(),
                            "model_thoughts": history.model_thoughts()
                        }
                    )

                except Exception as e:
                    await ws_handler.send_error(f"Error processing request: {str(e)}")
                    
        except WebSocketDisconnect:
            print(f"Client #{connection_id} disconnected")
        finally:
            # Always clean up the connection
            await state_manager.remove_connection(connection_id)
            
    except WebSocketException as e:
        print(f"WebSocket error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/agent/stop")
async def stop_agent(websocket: WebSocket):
    try:
        await websocket.accept()
        state_manager.request_stop()
        await websocket.send_json({"status": "success", "message": "Stop request sent"})
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.get("/agent/status")
async def get_agent_status():
    return {
        "browser_active": state_manager.state.browser is not None,
        "context_active": state_manager.state.browser_context is not None,
        "agent_active": state_manager.state.agent is not None,
        "active_connections": len(state_manager.state.active_connections),
        "stop_requested": state_manager.state.agent_state.is_stop_requested()
    }

@app.post("/agent/cleanup")
async def cleanup_resources(keep_browser_open: bool = False):
    await state_manager.cleanup(keep_browser_open)
    return {"status": "success", "message": "Resources cleaned up"} 