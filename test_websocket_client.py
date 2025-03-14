#!/usr/bin/env python3
import asyncio
import json
from websockets.client import connect
from websockets.exceptions import ConnectionClosed
from datetime import datetime
import argparse
import sys

async def connect_and_run_task(task, add_infos="", headless=False, max_steps=100):
    uri = "ws://13.58.59.22:8000/agent/ws"
    
    try:
        print(f"Connecting to {uri}...")
        async with connect(uri) as websocket:
            print("✅ Connected to WebSocket server")

            # Create task request
            task_request = {
                "task": task,
                "add_infos": add_infos,
                "config": {
                    "headless": headless,
                    "use_vision": True,
                    "max_steps": max_steps
                }
            }

            # Send the task
            print(f"\n📤 Sending task: {task_request['task']}")
            if add_infos:
                print(f"Additional info: {add_infos}")
            await websocket.send(json.dumps(task_request))

            # Receive and process messages
            while True:
                try:
                    message = await websocket.recv()
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
                        print(f"\n❌ {formatted_time}Error: {data['error']}")
                        if 'details' in data:
                            print(f"Details: {data['details']}")
                        break

                    msg_type = data.get('type')
                    msg_data = data.get('data', {})

                    if msg_type == 'status':
                        status = msg_data.get('status', 'unknown')
                        status_emoji = "🚀" if status == "starting" else "⏳" if status == "running" else "ℹ️"
                        print(f"\n{status_emoji} {formatted_time}Status: {status}")
                        if 'message' in msg_data:
                            print(f"Message: {msg_data['message']}")
                        if 'task_id' in msg_data:
                            print(f"Task ID: {msg_data['task_id']}")
                    
                    elif msg_type == 'log':
                        level = msg_data.get('level', 'INFO')
                        message = msg_data.get('message', '')
                        print(f"{formatted_time}{level}: {message}")
                    
                    elif msg_type == 'tool_usage':
                        step = msg_data.get('step', 0)
                        thought = msg_data.get('thought', '')
                        actions = msg_data.get('actions', [])
                        summary = msg_data.get('summary', '')
                        task_progress = msg_data.get('task_progress', '')
                        future_plans = msg_data.get('future_plans', '')
                        
                        print(f"\n🔄 {formatted_time}Step {step}:")
                        print(f"🧠 Thought: {thought}")
                        
                        if actions:
                            print("\n🛠️  Tools used:")
                            for i, action in enumerate(actions):
                                action_type = action.get('type', 'unknown')
                                params = action.get('parameters', {})
                                print(f"  {i+1}. {action_type}")
                                if params:
                                    for param_name, param_value in params.items():
                                        print(f"     - {param_name}: {param_value}")
                        
                        if summary:
                            print(f"\n📋 Summary: {summary}")
                        if task_progress:
                            print(f"\n📊 Progress: {task_progress}")
                        if future_plans:
                            print(f"\n🔮 Future plans: {future_plans}")
                        
                        print("\n" + "-" * 50)
                    
                    elif msg_type == 'result':
                        print("\n✅ === Final Results ===")
                        print(f"Result: {msg_data.get('final_result', '')}")
                        if msg_data.get('errors'):
                            print(f"\n❌ Errors: {msg_data['errors']}")
                        if msg_data.get('model_actions'):
                            print("\n🔍 Model Actions:")
                            print(msg_data['model_actions'])
                        if msg_data.get('model_thoughts'):
                            print("\n💭 Model Thoughts:")
                            print(msg_data['model_thoughts'])
                        break

                except ConnectionClosed:
                    print("\n🔌 Connection closed by server")
                    break
                except json.JSONDecodeError as e:
                    print(f"\n❌ Failed to parse JSON message: {e}")
                except Exception as e:
                    print(f"\n❌ Error processing message: {str(e)}")

    except ConnectionRefusedError:
        print("❌ Failed to connect to WebSocket server. Make sure the server is running.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="WebSocket Agent Client")
    parser.add_argument("--task", type=str, default="go into deepseek webpage check all api prices, do a table then go to claude check api prices and return me both tables of prices. '", 
                        help="The task for the agent to perform")
    parser.add_argument("--info", type=str, default="", 
                        help="Additional information for the agent")
    parser.add_argument("--headless", action="store_true", 
                        help="Run the browser in headless mode")
    parser.add_argument("--steps", type=int, default=200, 
                        help="Maximum number of steps for the agent")
    
    args = parser.parse_args()
    
    print("🚀 Starting WebSocket client test...")
    print(f"Task: {args.task}")
    if args.info:
        print(f"Additional info: {args.info}")
    print(f"Headless mode: {'Yes' if args.headless else 'No'}")
    print(f"Max steps: {args.steps}")
    print("-" * 50)
    
    try:
        asyncio.run(connect_and_run_task(
            task=args.task,
            add_infos=args.info,
            headless=args.headless,
            max_steps=args.steps
        ))
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()