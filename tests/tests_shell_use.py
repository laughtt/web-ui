# Using relative imports to go up one level and import from src
import sys
import os
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print(project_root)
from src.utils.shell import ShellTools
# Using relative imports to go up one level and import from src

print(project_root)
def test_shell_tools():
    print("===== Testing ShellTools Module =====\n")
    
    # Test 1: Basic command execution
    print("Test 1: Basic command execution")
    result = ShellTools.shell_exec("echo 'Hello, World!'")
    print(f"Result: {result}\n")
    
    # Test 2: Start an interactive process
    print("Test 2: Starting an interactive process")
    process_id = ShellTools.shell_exec("python -c \"while True: print(input('Enter text: '))\"", 
                                     interactive=True, 
                                     process_id="interactive_test")
    print(f"Process started: {process_id}\n")
    
    # Test 3: Write to the interactive process
    print("Test 3: Writing to the interactive process")
    write_result = ShellTools.shell_write_to_process("interactive_test", "Hello from LLM!")
    print(f"Write result: {write_result}\n")
    
    # Test 4: View process output
    print("Test 4: Viewing process output")
    view_result = ShellTools.shell_view("interactive_test")
    print(f"View result: {view_result}\n")
    
    # Test 5: Write again to the process
    print("Test 5: Writing again to the interactive process")
    write_result = ShellTools.shell_write_to_process("interactive_test", "Second message")
    print(f"Write result: {write_result}\n")
    
    # Test 6: View updated output
    print("Test 6: Viewing updated output")
    view_result = ShellTools.shell_view("interactive_test")
    print(f"View result: {view_result}\n")
    
    # Test 7: Kill the process
    print("Test 7: Killing the process")
    kill_result = ShellTools.shell_kill_process("interactive_test")
    print(f"Kill result: {kill_result}\n")
    
    # Test 8: Long-running process with wait
    print("Test 8: Testing shell_wait with a countdown")
    count_id = ShellTools.shell_exec("for i in 5 4 3 2 1; do echo \"Countdown: $i\"; sleep 1; done; echo \"Done!\"", 
                                   interactive=True,
                                   process_id="countdown")
    print(f"Started countdown: {count_id}")
    wait_result = ShellTools.shell_wait("countdown", timeout=10, check_interval=0.5)
    print(f"Wait result: {wait_result}\n")
    
    print("===== All tests completed =====")

if __name__ == "__main__":
    test_shell_tools()