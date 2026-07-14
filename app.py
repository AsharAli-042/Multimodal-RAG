import subprocess
import sys

steps = [
    "src/extract_text.py",
    "src/extract_images.py",
    "src/extract_tables.py",
    "src/vision.py",
    "src/embed.py",
    "src/vector_store.py",
    "src/rag.py"
]

for step in steps:

    print("=" * 70)
    print(f"Running {step}")
    print("=" * 70)

    result = subprocess.run([sys.executable, step])

    if result.returncode != 0:

        print(f"\nError while running {step}")

        break

print("\nPipeline Finished.")