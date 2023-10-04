import subprocess
import dLux as dl
from jax.tree_util import tree_flatten
from selenium import webdriver
import os


def get_parent_depth(cls):
    if cls.__bases__:
        return max(get_parent_depth(base) for base in cls.__bases__) + 1
    else:
        return 0


def save_to_png(html_file, image_file):
    # Set the options for the conversion
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    options.add_argument("disable-gpu")
    options.add_argument("no-sandbox")

    # Create a new Chrome driver
    driver = webdriver.Chrome(options=options)

    # Load the HTML file in the driver
    driver.get(f"file://{html_file}")

    # # Scroll to the bottom of the page
    height = driver.execute_script(
        "return Math.max( document.body.scrollHeight, document.body.offsetHeight, "
        "document.documentElement.clientHeight, document.documentElement.scrollHeight, "
        "document.documentElement.offsetHeight );"
    )

    # Set the size of the window to the height of the page
    driver.set_window_size(height, height)

    # Take a screenshot of the driver and save it to an image file
    driver.save_screenshot(image_file)

    # Quit the driver
    driver.quit()


classes = tree_flatten(dl.__all__)[0]
dl_dict = dl.__dict__
paths = [str(dl_dict[c]).split("'")[1] for c in classes]
depths = [get_parent_depth(dl_dict[c]) for c in classes]
cwd = os.getcwd()

# Generate UMLS for each class
for path, depth in zip(paths, depths):
    # print(depth, path)

    # Generate UML
    subprocess.run(
        [
            "pyreverse",
            "-c",
            f"{path}",
            "--output",
            "html",
            "-s0",
            f"-a{depth-3}",
            "--colorized",
            "dLux",
            "--output-directory",
            "assets/uml",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Save to png
    file_name = path.split(".")[-1]
    save_to_png(
        f"{cwd}/assets/uml/{path}.html", f"{cwd}/assets/uml/{file_name}.png"
    )

    # Remove html file
    os.remove(f"{cwd}/assets/uml/{path}.html")

# # Generate for whole package
# subprocess.run(
#     [
#         "pyreverse",
#         "-o",
#         "html",
#         "--colorized",
#         "dLux",
#         "--output-directory",
#         "uml_files",
#     ],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
# )


"""
    ??? abstract "UML"
        ![UML](../../assets/uml/Optic.png)
"""
