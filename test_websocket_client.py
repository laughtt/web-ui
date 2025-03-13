#!/usr/bin/env python3
import asyncio
import json
from websockets.client import connect
from websockets.exceptions import ConnectionClosed
from datetime import datetime

async def connect_and_run_task():
    uri = "ws://localhost:8000/agent/ws"
    
    try:
        async with connect(uri) as websocket:
            print("Connected to WebSocket server")

            # Example task
            task_request = {
                "task": "go to google.com and search for 'OpenAI'",
                "add_infos": "Click on the first search result",
                "config": {
                    "headless": False,
                    "use_vision": True,
                    "max_steps": 10
                }
            }

            # Send the task
            print(f"\nSending task: {task_request['task']}")
            await websocket.send(json.dumps(task_request))

            # Receive and process messages
            while True:
                try:
                    message = await websocket.recv()
                    print(f"\nRaw message: {message}")  # Debug print
                    data = json.loads(message)
                    
                    # Get timestamp if available
                    formatted_time = ""
                    if 'timestamp' in data:
                        try:
                            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                            formatted_time = f"[{timestamp.strftime('%H:%M:%S')}] "
                        except (ValueError, AttributeError):
                            pass

                    # Process different message types
                    if 'error' in data:
                        print(f"\n{formatted_time}Error: {data['error']}")
                        if 'details' in data:
                            print(f"Details: {data['details']}")
                        break

                    msg_type = data.get('type')
                    msg_data = data.get('data', {})

                    if msg_type == 'status':
                        print(f"\n{formatted_time}Status: {msg_data.get('status', 'unknown')}")
                        if 'message' in msg_data:
                            print(f"Message: {msg_data['message']}")
                    
                    elif msg_type == 'log':
                        level = msg_data.get('level', 'INFO')
                        message = msg_data.get('message', '')
                        print(f"{formatted_time}{level}: {message}")
                    
                    elif msg_type == 'result':
                        print("\n=== Final Results ===")
                        print(f"Result: {msg_data.get('final_result', '')}")
                        if msg_data.get('errors'):
                            print(f"\nErrors: {msg_data['errors']}")
                        if msg_data.get('model_actions'):
                            print("\nModel Actions:")
                            print(msg_data['model_actions'])
                        if msg_data.get('model_thoughts'):
                            print("\nModel Thoughts:")
                            print(msg_data['model_thoughts'])
                        break

                except ConnectionClosed:
                    print("\nConnection closed by server")
                    break
                except json.JSONDecodeError as e:
                    print(f"\nFailed to parse JSON message: {e}")
                except Exception as e:
                    print(f"\nError processing message: {str(e)}")

    except ConnectionRefusedError:
        print("Failed to connect to WebSocket server. Make sure the server is running.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Starting WebSocket client test...")
    asyncio.run(connect_and_run_task())