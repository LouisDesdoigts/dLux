import os

# Set the directory containing the .ipynb files
notebook_dir = "tutorials/"

# Recursively find all .ipynb files in the directory
notebooks = []
for root, dirs, files in os.walk(notebook_dir):
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(root, file))

# Convert each notebook to .md using jupyter nbconvert
for notebook in notebooks:
    md_path = os.path.join(
        os.path.dirname(os.path.dirname(notebook)),
        os.path.splitext(os.path.basename(notebook))[0] + ".md",
    )
    output_dir = os.path.dirname(os.path.dirname(notebook))
    os.system(
        (
            f"jupyter nbconvert --to Markdown {notebook} --output-dir {output_dir}"
            f" --output {os.path.basename(md_path)}"
        )
    )
