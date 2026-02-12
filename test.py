import ray
import os

# Read the head node IP
with open('ray_utils/ray_head_ip.txt', 'r') as f:
    head_ip = f.read().strip()

# Get the port from environment or use default
ray_port = os.environ.get('MASTER_PORT', '6379')

# Connect explicitly to the head node
ray.init(address=f'{head_ip}:{ray_port}')

@ray.remote
def test_task():
    import socket
    return f"Hello from {socket.gethostname()}"

result = ray.get(test_task.remote())
print(result)
