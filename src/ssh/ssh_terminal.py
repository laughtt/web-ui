import asyncio
import subprocess
import threading
import re
import time
import os
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
            
    def get_screen_state(self) -> str:
        """Get the current screen state as a string."""
        return '\n'.join([''.join(line) for line in self.buffer])
    
    def get_history(self) -> List[str]:
        """Get terminal history."""
        return self.history


class LocalConnection:
    """Executes commands locally on the server."""
    
    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir
        self.terminal_state = TerminalState()
        self.connected = False
        self._output_callback = None
        self._process = None
        self._reading_thread = None
        self._stop_reading = threading.Event()
        
    def connect(self) -> None:
        """Initialize the local connection."""
        # Create a shell process
        shell_command = 'bash' if os.name != 'nt' else 'cmd.exe'
        
        self._process = subprocess.Popen(
            shell_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            text=True,
            bufsize=1,
            cwd=self.working_dir,
            universal_newlines=True
        )
        
        self.connected = True
        
        # Start background thread to read output
        self._stop_reading.clear()
        self._reading_thread = threading.Thread(target=self._read_output)
        self._reading_thread.daemon = True
        self._reading_thread.start()
        
    def disconnect(self) -> None:
        """Close the local connection."""
        if self.connected:
            self._stop_reading.set()
            if self._reading_thread:
                self._reading_thread.join(timeout=2)
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self.connected = False
    
    def execute_command(self, command: str) -> None:
        """Execute a command locally."""
        if not self.connected:
            raise RuntimeError("Terminal not connected")
        
        # Send command with newline to execute
        if self._process and self._process.stdin:
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
    
    def _read_output(self) -> None:
        """Background thread to read process output."""
        while not self._stop_reading.is_set() and self.connected:
            if self._process and self._process.stdout:
                # Read one line (or as much as is available)
                try:
                    line = self._process.stdout.readline()
                    if line:
                        self.terminal_state.process_output(line)
                        
                        # Notify callback if set
                        if self._output_callback:
                            self._output_callback(line)
                    else:
                        # Process may have ended
                        time.sleep(0.1)
                except Exception as e:
                    if self._output_callback:
                        self._output_callback(f"Error reading output: {str(e)}\n")
                    time.sleep(0.1)
            else:
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


class LocalTerminal:
    """Local terminal manager class that handles multiple terminal sessions."""
    
    def __init__(self):
        self.connections: Dict[str, LocalConnection] = {}
    
    def connect(self, name: str, working_dir: Optional[str] = None) -> LocalConnection:
        """Create and establish a new local terminal connection."""
        connection = LocalConnection(working_dir=working_dir)
        connection.connect()
        self.connections[name] = connection
        return connection
    
    def disconnect(self, name: str) -> None:
        """Disconnect and remove a named terminal connection."""
        if name in self.connections:
            self.connections[name].disconnect()
            del self.connections[name]
    
    def disconnect_all(self) -> None:
        """Disconnect all terminal connections."""
        for name in list(self.connections.keys()):
            self.disconnect(name)
    
    def get_connection(self, name: str) -> LocalConnection:
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