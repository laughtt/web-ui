import subprocess
import os
import signal
import time
import sys
from typing import Optional, Dict, List, Union, Tuple

class ShellTools:
    """A module for LLM-based shell interaction tools."""
    
    _processes: Dict[str, subprocess.Popen] = {}
    
    @classmethod
    def shell_exec(cls, 
                   command: str, 
                   process_id: Optional[str] = None, 
                   cwd: Optional[str] = None, 
                   env: Optional[Dict[str, str]] = None,
                   interactive: bool = False) -> str:
        """
        Executes commands in a shell session.
        
        Args:
            command: The command to execute
            process_id: Optional identifier for the process (needed for interactive processes)
            cwd: Optional working directory for the command
            env: Optional environment variables
            interactive: Whether this is an interactive process that will receive input
            
        Returns:
            Process ID if interactive, otherwise command output
        """
        if not command:
            return "Error: No command provided"
        
        # Generate a process ID if not provided
        if not process_id and interactive:
            process_id = f"process_{len(cls._processes) + 1}"
        
        try:
            if interactive:
                # Start an interactive process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    env=env,
                    universal_newlines=True
                )
                cls._processes[process_id] = process
                return f"Process started with ID: {process_id}"
            else:
                # Run a non-interactive command and return the output
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    env=env
                )
                output = result.stdout
                error = result.stderr
                
                if error:
                    return f"Output:\n{output}\n\nErrors:\n{error}"
                return output.strip()
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    @classmethod
    def shell_view(cls, process_id: str, timeout: float = 0.1) -> str:
        """
        Displays output from a shell session.
        
        Args:
            process_id: The ID of the process to view
            timeout: Time to wait for output before returning
            
        Returns:
            The output from the process
        """
        if process_id not in cls._processes:
            return f"Error: No process found with ID {process_id}"
        
        process = cls._processes[process_id]
        output = ""
        
        try:
            # Check if there's any output available
            if process.stdout.readable():
                # Use a non-blocking approach to read available output
                import select
                if select.select([process.stdout], [], [], timeout)[0]:
                    # Read available output without blocking
                    output_lines = []
                    while select.select([process.stdout], [], [], 0)[0]:
                        line = process.stdout.readline()
                        if not line:
                            break
                        output_lines.append(line)
                    output = "".join(output_lines)
                
                # Check for errors
                if select.select([process.stderr], [], [], timeout)[0]:
                    error_lines = []
                    while select.select([process.stderr], [], [], 0)[0]:
                        line = process.stderr.readline()
                        if not line:
                            break
                        error_lines.append(line)
                    
                    if error_lines:
                        error = "".join(error_lines)
                        if output:
                            output += f"\n\nErrors:\n{error}"
                        else:
                            output = f"Errors:\n{error}"
            
            # Check if process has terminated
            if process.poll() is not None:
                exit_code = process.poll()
                if output:
                    output += f"\n\nProcess exited with code: {exit_code}"
                else:
                    output = f"Process exited with code: {exit_code}"
                # Clean up the process
                cls._processes.pop(process_id, None)
                
            return output if output else "No output available"
        except Exception as e:
            return f"Error viewing process output: {str(e)}"
    
    @classmethod
    def shell_wait(cls, 
                  process_id: str, 
                  timeout: Optional[float] = None, 
                  check_interval: float = 0.5) -> str:
        """
        Waits for a process to complete within a shell session.
        
        Args:
            process_id: The ID of the process to wait for
            timeout: Maximum time to wait (in seconds)
            check_interval: How often to check/report status
            
        Returns:
            Final output from the process
        """
        if process_id not in cls._processes:
            return f"Error: No process found with ID {process_id}"
        
        process = cls._processes[process_id]
        start_time = time.time()
        output_chunks = []
        
        try:
            while process.poll() is None:
                # Check for timeout
                if timeout and (time.time() - start_time) > timeout:
                    return f"Timeout after {timeout} seconds. Process is still running."
                
                # Get any new output
                new_output = cls.shell_view(process_id, check_interval)
                if new_output and new_output != "No output available":
                    output_chunks.append(new_output)
                
                time.sleep(check_interval)
            
            # Process has completed, get final output
            final_output = cls.shell_view(process_id, 0)
            if final_output and final_output != "No output available":
                output_chunks.append(final_output)
            
            # Clean up the process
            exit_code = process.returncode
            cls._processes.pop(process_id, None)
            
            all_output = "\n".join(output_chunks)
            return f"{all_output}\n\nProcess completed with exit code: {exit_code}"
        except Exception as e:
            return f"Error waiting for process: {str(e)}"
    
    @classmethod
    def shell_write_to_process(cls, process_id: str, input_text: str) -> str:
        """
        Sends input to interactive processes.
        
        Args:
            process_id: The ID of the process to send input to
            input_text: The text to send to the process
            
        Returns:
            Status message
        """
        if process_id not in cls._processes:
            return f"Error: No process found with ID {process_id}"
        
        process = cls._processes[process_id]
        
        try:
            # Check if process is still running
            if process.poll() is not None:
                return f"Error: Process {process_id} has already exited"
            
            # Ensure input ends with newline
            if not input_text.endswith('\n'):
                input_text += '\n'
            
            # Write to process stdin
            process.stdin.write(input_text)
            process.stdin.flush()
            
            # Give the process time to process the input
            time.sleep(0.1)
            
            return f"Input sent to process {process_id}"
        except Exception as e:
            return f"Error sending input to process: {str(e)}"
    
    @classmethod
    def shell_kill_process(cls, process_id: str, force: bool = False) -> str:
        """
        Terminates running processes.
        
        Args:
            process_id: The ID of the process to terminate
            force: Whether to force termination (SIGKILL instead of SIGTERM)
            
        Returns:
            Status message
        """
        if process_id not in cls._processes:
            return f"Error: No process found with ID {process_id}"
        
        process = cls._processes[process_id]
        
        try:
            # Check if process is still running
            if process.poll() is not None:
                cls._processes.pop(process_id, None)
                return f"Process {process_id} has already exited"
            
            # Send signal to terminate process
            if os.name == 'nt':  # Windows
                if force:
                    process.kill()
                else:
                    process.terminate()
            else:  # Unix-like
                if force:
                    process.send_signal(signal.SIGKILL)
                else:
                    process.send_signal(signal.SIGTERM)
            
            # Wait a bit for the process to terminate
            process.wait(timeout=2)
            
            # Clean up
            cls._processes.pop(process_id, None)
            
            return f"Process {process_id} {'forcefully ' if force else ''}terminated"
        except subprocess.TimeoutExpired:
            if force:
                process.kill()
                cls._processes.pop(process_id, None)
                return f"Process {process_id} forcefully terminated after timeout"
            else:
                return f"Process {process_id} did not terminate gracefully, try force=True"
        except Exception as e:
            return f"Error terminating process: {str(e)}"