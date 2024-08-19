import subprocess

# List of packages we want to include
packages = [
    "click",
    "pyzmq",
    "grpcio-tools",
    "pytest",
    "FlagEmbedding",
    "pydantic",
    "peft"
]

# Run pip freeze and capture the output
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

# Filter the output to include only our specified packages
requirements = []
for line in result.stdout.split('\n'):
    for package in packages:
        if line.lower().startswith(package.lower()):
            requirements.append(line)
            break

# Write the filtered requirements to a file
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(requirements))

print("requirements.txt has been generated.")