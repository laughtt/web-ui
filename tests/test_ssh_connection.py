#!/usr/bin/env python3
import asyncio
import json
import websockets
import argparse
import getpass
from datetime import datetime
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def ssh_terminal_test(server_url, ssh_host, ssh_username, ssh_password=None, ssh_port=22):
    """
    Test the SSH Terminal WebSocket endpoint.
    
    Args:
        server_url: The WebSocket server URL (e.g., ws://localhost:8000/ssh/ws)
        ssh_host: The SSH server hostname or IP
        ssh_username: The SSH username
        ssh_password: The SSH password (will prompt if not provided)
        ssh_port: The SSH port (defaults to 22)
    """

    logger.info(f"Connecting to WebSocket server at {server_url}...")
    
    connection_name = None
    
    try:
        # Add timeout for initial connection
        async with websockets.connect(server_url, ping_interval=20, ping_timeout=30) as websocket:
            logger.info("WebSocket connection established successfully.")
            
            # Handle incoming messages in a separate task
            async def receive_messages():
                logger.info("Starting message receiver task")
                while True:
                    try:
                        logger.debug("Waiting for message...")
                        message = await websocket.recv()
                        logger.debug(f"Received raw message: {message[:100]}...")
                        
                        data = json.loads(message)
                        
                        timestamp = data.get("timestamp", datetime.now().isoformat())
                        msg_type = data.get("type", "unknown")
                        logger.debug(f"Received message type: {msg_type}")
                        
                        if msg_type == "connection_confirmed":
                            logger.info(f"✅ Connection confirmed: {data['data']['message']}")
                            logger.info(f"Client ID: {data['data']['client_id']}")
                            
                            # Now connect to the SSH server
                            logger.info(f"Connecting to SSH server {ssh_host}...")
                            connect_payload = {
                                "type": "connect",
                                "connection_name": "test_connection",
                                "connection": {
                                    "host": ssh_host,
                                    "username": ssh_username,
                                    "password": ssh_password,
                                    "port": ssh_port
                                }
                            }
                            logger.debug(f"Sending connect payload: {json.dumps(connect_payload, default=str)}")
                            await websocket.send(json.dumps(connect_payload))
                            
                        elif msg_type == "ssh_connected":
                            logger.info(f"✅ SSH Connected: {data['data']['message']}")
                            nonlocal connection_name
                            connection_name = data['data']['connection_name']
                            logger.info("Type commands to send to the SSH server. Type 'exit' to quit.")
                            
                        elif msg_type == "terminal_output":
                            # Print terminal output with connection name
                            conn = data['data']['connection_name']
                            output = data['data']['output']
                            print(f"\033[93m[{conn}]\033[0m {output}", end="", flush=True)
                            
                        elif msg_type == "terminal_state":
                            logger.info(f"\n--- Terminal State for {data['data']['connection_name']} ---")
                            print(data['data']['state'])
                            logger.info("-----------------------------------")
                            
                        elif msg_type == "ssh_disconnected":
                            logger.info(f"SSH connection {data['data']['connection_name']} disconnected")
                            
                        elif msg_type == "ssh_error":
                            logger.error(f"Error: {data['data'].get('error', 'Unknown error')}")
                            if 'details' in data['data']:
                                logger.error(f"Details: {data['data']['details']}")
                                
                        elif msg_type == "error":
                            logger.error(f"WebSocket Error: {data.get('error', 'Unknown error')}")
                            if 'details' in data:
                                logger.error(f"Details: {data['details']}")
                                
                        elif msg_type == "command_sent":
                            # Command confirmation received
                            logger.debug(f"Command sent: {data['data'].get('command', '')}")
                            
                        elif msg_type == "pong":
                            # Pong response
                            logger.debug("Received pong response")
                            
                        else:
                            logger.info(f"Received unknown message type: {msg_type}")
                            logger.debug(f"Full message: {data}")
                            
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"Connection to server closed: {e}")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"Raw message: {message[:200]}...")
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving message: {str(e)}")
                        logger.error(traceback.format_exc())
                        break
                
                logger.info("Message receiver task ended")
            
            # Start the message receiver
            logger.info("Creating receiver task")
            receiver_task = asyncio.create_task(receive_messages())
            
            # Send an initial ping to ensure communication is working
            logger.info("Sending initial ping")
            await websocket.send(json.dumps({"type": "ping"}))
            
            try:
                # Main command input loop
                logger.info("Starting main command loop")
                while True:
                    # Wait for connection_name to be set before accepting commands
                    if not connection_name:
                        logger.debug("Waiting for SSH connection to be established...")
                        await asyncio.sleep(1)
                        
                        # Periodically send ping to keep connection alive
                        try:
                            await websocket.send(json.dumps({"type": "ping"}))
                            logger.debug("Ping sent")
                        except Exception as e:
                            logger.error(f"Error sending ping: {e}")
                        
                        continue
                    
                    try:
                        command = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: input("\033[92m> \033[0m")
                        )
                        
                        if not command:
                            continue
                            
                        if command.lower() == 'exit':
                            logger.info("Disconnecting...")
                            await websocket.send(json.dumps({
                                "type": "disconnect",
                                "connection_name": connection_name
                            }))
                            break
                        
                        elif command.lower() == 'state':
                            # Get current terminal state
                            logger.info("Requesting terminal state...")
                            await websocket.send(json.dumps({
                                "type": "get_state",
                                "connection_name": connection_name
                            }))
                            
                        elif command.lower() == 'ping':
                            # Send a ping
                            logger.info("Sending ping...")
                            await websocket.send(json.dumps({"type": "ping"}))
                            
                        else:
                            # Send command to the SSH server
                            logger.debug(f"Sending command: {command}")
                            await websocket.send(json.dumps({
                                "type": "execute",
                                "connection_name": connection_name,
                                "command": command
                            }))
                            
                    except (EOFError, KeyboardInterrupt):
                        logger.info("Exiting due to user interrupt...")
                        break
            
            finally:
                # Cancel the receiver task
                logger.info("Canceling receiver task")
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    logger.info("Receiver task canceled")
                
                # Ensure disconnect before exiting
                if connection_name:
                    try:
                        logger.info(f"Disconnecting from {connection_name}...")
                        await websocket.send(json.dumps({
                            "type": "disconnect",
                            "connection_name": connection_name
                        }))
                        logger.info("Disconnected from SSH server.")
                    except Exception as e:
                        logger.error(f"Error during disconnect: {e}")
                
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"General error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Test completed")

def main():
    parser = argparse.ArgumentParser(description="Test SSH Terminal WebSocket")
    parser.add_argument("--server", default="ws://localhost:8000/ssh/ws", 
                        help="WebSocket server URL")
    parser.add_argument("--ip", required=True, help="SSH server IP address")
    parser.add_argument("--port", type=int, default=8000, help="SSH server port")
    parser.add_argument("--user", required=True, help="SSH username")
    parser.add_argument("--password", help="SSH password (will prompt if not provided)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        logger.info("Starting SSH terminal test")
        asyncio.run(ssh_terminal_test(
            args.server, 
            args.ip, 
            args.user, 
            args.password, 
            args.port
        ))
    except KeyboardInterrupt:
        logger.info("Test terminated by user.")
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()