import asyncio
import paramiko
import threading
import re
import time
from typing import Dict, List, Optional, Callable, Any, Union, Tuple

class TerminalState:
    """Tracks the state of a virtual terminal including screen buffer and cursor position."""
    
    def __init__(self, rows: int = 24, cols: int = 80):
        self.rows = rows
        self.cols = cols
        self.buffer: List[List[str]] = [[''] * cols for _ in range(rows)]
        self.cursor_row = 0
        self.cursor_col = 0
        self.ansi_escape_pattern = re.compile(r'\x1b\[([0-9;]*)([a-zA-Z])')
        self.history: List[str] = []
        
    def process_output(self, output: str) -> None:
        """Process terminal output including ANSI escape sequences."""
        # Split the output by escape sequences
        parts = self.ansi_escape_pattern.split(output)
        
        i = 0
        while i < len(parts):
            if i == 0 and parts[i]:
                # First part is always text
                self._write_text(parts[i])
                i += 1
            elif i + 2 < len(parts):
                # Process escape sequence: parts[i] is params, parts[i+1] is command
                params = parts[i] or ''
                command = parts[i+1]
                self._process_escape_sequence(params, command)
                
                # Next part is text
                if i + 2 < len(parts) and parts[i+2]:
                    self._write_text(parts[i+2])
                i += 3
            else:
                # Something unexpected happened
                break
                
    def _write_text(self, text: str) -> None:
        """Write text to the terminal at current cursor position."""
        for char in text:
            if char == '\n':
                self.history.append(''.join(self.buffer[self.cursor_row]))
                self.cursor_row += 1
                self.cursor_col = 0
                # Scroll if needed
                if self.cursor_row >= self.rows:
                    self.buffer.pop(0)
                    self.buffer.append([''] * self.cols)
                    self.cursor_row = self.rows - 1
            elif char == '\r':
                self.cursor_col = 0
            else:
                if self.cursor_col < self.cols:
                    self.buffer[self.cursor_row][self.cursor_col] = char
                    self.cursor_col += 1
    
    def _process_escape_sequence(self, params: str, command: str) -> None:
        """Process ANSI escape sequence."""
        if command == 'm':
            # Color/style commands - we could implement if needed
            pass
        elif command == 'A':
            # Cursor up
            steps = int(params) if params else 1
            self.cursor_row = max(0, self.cursor_row - steps)
        elif command == 'B':
            # Cursor down
            steps = int(params) if params else 1
            self.cursor_row = min(self.rows - 1, self.cursor_row + steps)
        elif command == 'C':
            # Cursor forward
            steps = int(params) if params else 1
            self.cursor_col = min(self.cols - 1, self.cursor_col + steps)
        elif command == 'D':
            # Cursor backward
            steps = int(params) if params else 1
            self.cursor_col = max(0, self.cursor_col - steps)
        # Many more ANSI commands could be implemented
            
    def get_screen_state(self) -> str:
        """Get the current screen state as a string."""
        return '\n'.join([''.join(line) for line in self.buffer])
    
    def get_history(self) -> List[str]:
        """Get terminal history."""
        return self.history


class SSHConnection:
    """Handles a single SSH connection to a remote server."""
    
    def __init__(self, host: str, username: str, 
                 password: Optional[str] = None, 
                 key_filename: Optional[str] = None,
                 port: int = 22):
        self.host = host
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.port = port
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.terminal_state = TerminalState()
        self.shell = None
        self.connected = False
        self._output_callback = None
        self._reading_thread = None
        self._stop_reading = threading.Event()
        
    def connect(self) -> None:
        """Establish SSH connection and start a shell."""
        self.client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_filename=self.key_filename
        )
        self.shell = self.client.invoke_shell()
        self.connected = True
        
        # Start background thread to read output
        self._stop_reading.clear()
        self._reading_thread = threading.Thread(target=self._read_output)
        self._reading_thread.daemon = True
        self._reading_thread.start()
        
    def disconnect(self) -> None:
        """Close the SSH connection."""
        if self.connected:
            self._stop_reading.set()
            if self._reading_thread:
                self._reading_thread.join(timeout=2)
            if self.shell:
                self.shell.close()
            self.client.close()
            self.connected = False
    
    def execute_command(self, command: str) -> None:
        """Execute a command in the SSH shell."""
        if not self.connected:
            raise RuntimeError("Not connected to SSH server")
        
        # Send command with newline to execute
        self.shell.send(command + "\n")
    
    def _read_output(self) -> None:
        """Background thread to read shell output."""
        while not self._stop_reading.is_set() and self.connected:
            if self.shell and self.shell.recv_ready():
                output = self.shell.recv(4096).decode('utf-8', errors='replace')
                self.terminal_state.process_output(output)
                
                # Notify callback if set
                if self._output_callback:
                    self._output_callback(output)
            time.sleep(0.1)
    
    def set_output_callback(self, callback: Callable[[str], Any]) -> None:
        """Set callback function to be called when new output is available."""
        self._output_callback = callback
    
    def get_terminal_state(self) -> str:
        """Get the current terminal state."""
        return self.terminal_state.get_screen_state()
    
    def get_terminal_history(self) -> List[str]:
        """Get the terminal command history."""
        return self.terminal_state.get_history()


class SSHTerminal:
    """Main SSH terminal manager class that handles multiple connections."""
    
    def __init__(self):
        self.connections: Dict[str, SSHConnection] = {}
    
    def connect(self, name: str, host: str, username: str, 
                password: Optional[str] = None,
                key_filename: Optional[str] = None, 
                port: int = 22) -> SSHConnection:
        """Create and establish a new SSH connection."""
        connection = SSHConnection(
            host=host,
            username=username,
            password=password,
            key_filename=key_filename,
            port=port
        )
        connection.connect()
        self.connections[name] = connection
        return connection
    
    def disconnect(self, name: str) -> None:
        """Disconnect and remove a named SSH connection."""
        if name in self.connections:
            self.connections[name].disconnect()
            del self.connections[name]
    
    def disconnect_all(self) -> None:
        """Disconnect all SSH connections."""
        for name in list(self.connections.keys()):
            self.disconnect(name)
    
    def get_connection(self, name: str) -> SSHConnection:
        """Get a connection by name."""
        if name not in self.connections:
            raise KeyError(f"No connection named '{name}'")
        return self.connections[name]
    
    def execute_command(self, name: str, command: str) -> None:
        """Execute a command on a specific connection."""
        self.get_connection(name).execute_command(command)
    
    def get_terminal_state(self, name: str) -> str:
        """Get the terminal state for a specific connection."""
        return self.get_connection(name).get_terminal_state()


# Add async support with asyncio
class AsyncSSHTerminal:
    """Asynchronous SSH terminal manager using asyncio."""
    
    def __init__(self):
        self.connections = {}
        self.tasks = {}
    
    async def connect(self, name: str, host: str, username: str,
                     password: Optional[str] = None,
                     key_filename: Optional[str] = None,
                     port: int = 22) -> None:
        """Asynchronously connect to an SSH server."""
        # Using asyncssh for better async support
        import asyncssh
        
        self.connections[name] = await asyncssh.connect(
            host=host,
            port=port,
            username=username,
            password=password,
            client_keys=key_filename
        )
        
        # Start a process to handle the shell
        self.tasks[name] = asyncio.create_task(
            self._handle_connection(name)
        )
    
    async def _handle_connection(self, name: str) -> None:
        """Manage an active connection."""
        conn = self.connections[name]
        process = await conn.start_shell()
        
        # Process stdin/stdout until closed
        while not process.stdout.at_eof():
            line = await process.stdout.readline()
            # Process the output and update terminal state
            # (We'd need to implement async terminal state handling)
    
    async def execute_command(self, name: str, command: str) -> str:
        """Execute a command and return its output."""
        conn = self.connections[name]
        result = await conn.run(command)
        return result.stdout
    
    async def disconnect(self, name: str) -> None:
        """Disconnect a specific connection."""
        if name in self.tasks:
            self.tasks[name].cancel()
            try:
                await self.tasks[name]
            except asyncio.CancelledError:
                pass
            del self.tasks[name]
        
        if name in self.connections:
            self.connections[name].close()
            del self.connections[name]
    
    async def disconnect_all(self) -> None:
        """Disconnect all connections."""
        for name in list(self.connections.keys()):
            await self.disconnect(name)