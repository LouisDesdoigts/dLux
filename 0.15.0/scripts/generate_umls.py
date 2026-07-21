import argparse
import inspect
import subprocess
import dLux as dl
from jax.tree_util import tree_flatten
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import sys
from pathlib import Path
from selenium.common.exceptions import WebDriverException

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
UML_DIR = DOCS_ROOT / "assets" / "uml"
PYREVERSE_TIMEOUT_SECONDS = 30
PAGE_LOAD_TIMEOUT_SECONDS = 20
SCREENSHOT_BUFFER_PX = 40
MIN_SCREENSHOT_WIDTH = 400
MIN_SCREENSHOT_HEIGHT = 300
MAX_SCREENSHOT_WIDTH = 12000
MAX_SCREENSHOT_HEIGHT = 12000


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
    options.add_argument("hide-scrollbars")

    # Create a new Chrome driver
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SECONDS)

        # Load the HTML file in the driver
        driver.get(f"file://{html_file}")

        # Hide scrollbars and normalize margins so screenshots only contain UML
        # content.
        driver.execute_script(
            "document.documentElement.style.overflow = 'hidden';"
            "document.body.style.overflow = 'hidden';"
            "document.documentElement.style.margin = '0';"
            "document.body.style.margin = '0';"
        )

        # Size to rendered UML content plus a small visual buffer.
        bbox_js = (
            "const body = document.body;"
            "const doc = document.documentElement;"
            "const svgs = Array.from(document.querySelectorAll('svg'))"
            ".filter(el => el.getBoundingClientRect().width > 0 && "
            "el.getBoundingClientRect().height > 0);"
            "let svgLeft = Infinity; let svgTop = Infinity;"
            "let svgRight = -Infinity; let svgBottom = -Infinity;"
            "for (const svg of svgs) {"
            "  const r = svg.getBoundingClientRect();"
            "  svgLeft = Math.min(svgLeft, r.left + window.scrollX);"
            "  svgTop = Math.min(svgTop, r.top + window.scrollY);"
            "  svgRight = Math.max(svgRight, r.right + window.scrollX);"
            "  svgBottom = Math.max(svgBottom, r.bottom + window.scrollY);"
            "}"
            "const docWidth = Math.max(body.scrollWidth, body.offsetWidth, "
            "doc.clientWidth, doc.scrollWidth, doc.offsetWidth);"
            "const docHeight = Math.max(body.scrollHeight, body.offsetHeight, "
            "doc.clientHeight, doc.scrollHeight, doc.offsetHeight);"
            "let left = 0; let top = 0; let width = docWidth; let height = docHeight;"
            "if (svgs.length) {"
            "  left = svgLeft;"
            "  top = svgTop;"
            "  width = Math.max(0, svgRight - svgLeft);"
            "  height = Math.max(0, svgBottom - svgTop);"
            "}"
            "return [Math.floor(left), Math.floor(top), Math.ceil(width), "
            "Math.ceil(height)];"
        )

        _, _, bbox_width, bbox_height = driver.execute_script(bbox_js)
        target_width, target_height = bbox_width + (
            2 * SCREENSHOT_BUFFER_PX
        ), bbox_height + (2 * SCREENSHOT_BUFFER_PX)
        target_width = int(
            min(
                max(target_width + SCREENSHOT_BUFFER_PX, MIN_SCREENSHOT_WIDTH),
                MAX_SCREENSHOT_WIDTH,
            )
        )
        target_height = int(
            min(
                max(target_height + SCREENSHOT_BUFFER_PX, MIN_SCREENSHOT_HEIGHT),
                MAX_SCREENSHOT_HEIGHT,
            )
        )

        # set_window_size uses outer window dimensions; convert desired inner
        # size to outer.
        chrome_dx, chrome_dy = driver.execute_script(
            "return [window.outerWidth - window.innerWidth, "
            "window.outerHeight - window.innerHeight];"
        )
        driver.set_window_size(
            target_width + int(chrome_dx), target_height + int(chrome_dy)
        )

        # Re-measure after first resize to catch late layout changes; then
        # resize once more.
        _, _, bbox_width_2, bbox_height_2 = driver.execute_script(bbox_js)
        target_width_2, target_height_2 = (
            bbox_width_2 + (2 * SCREENSHOT_BUFFER_PX),
            bbox_height_2 + (2 * SCREENSHOT_BUFFER_PX),
        )
        target_width_2 = int(
            min(
                max(target_width_2 + SCREENSHOT_BUFFER_PX, MIN_SCREENSHOT_WIDTH),
                MAX_SCREENSHOT_WIDTH,
            )
        )
        target_height_2 = int(
            min(
                max(target_height_2 + SCREENSHOT_BUFFER_PX, MIN_SCREENSHOT_HEIGHT),
                MAX_SCREENSHOT_HEIGHT,
            )
        )
        if target_width_2 != target_width or target_height_2 != target_height:
            chrome_dx_2, chrome_dy_2 = driver.execute_script(
                "return [window.outerWidth - window.innerWidth, "
                "window.outerHeight - window.innerHeight];"
            )
            driver.set_window_size(
                target_width_2 + int(chrome_dx_2),
                target_height_2 + int(chrome_dy_2),
            )

        # Screenshot a temporary element matching UML bounds to avoid viewport
        # minimum-width whitespace.
        bbox_left, bbox_top, bbox_width, bbox_height = driver.execute_script(bbox_js)
        capture_left = max(0, int(bbox_left - SCREENSHOT_BUFFER_PX))
        capture_top = max(0, int(bbox_top - SCREENSHOT_BUFFER_PX))
        capture_width = max(1, int(bbox_width + (2 * SCREENSHOT_BUFFER_PX)))
        capture_height = max(1, int(bbox_height + (2 * SCREENSHOT_BUFFER_PX)))

        driver.execute_script(
            "const id='__uml_capture_box__';"
            "const old=document.getElementById(id); if (old) old.remove();"
            "const box=document.createElement('div');"
            "box.id=id;"
            "box.style.position='absolute';"
            "box.style.left=arguments[0] + 'px';"
            "box.style.top=arguments[1] + 'px';"
            "box.style.width=arguments[2] + 'px';"
            "box.style.height=arguments[3] + 'px';"
            "box.style.background='transparent';"
            "box.style.pointerEvents='none';"
            "box.style.zIndex='2147483647';"
            "document.body.appendChild(box);"
            "window.scrollTo(arguments[0], arguments[1]);",
            capture_left,
            capture_top,
            capture_width,
            capture_height,
        )

        driver.find_element(By.ID, "__uml_capture_box__").screenshot(image_file)
        driver.execute_script(
            "const el=document.getElementById('__uml_capture_box__'); "
            "if (el) el.remove();"
        )
    finally:
        # Quit the driver
        driver.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UML PNGs for dLux classes.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional fully-qualified class paths to generate. "
            "If omitted, all discovered classes are generated."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of classes to process after filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dl_dict = dl.__dict__
    classes = [
        name
        for name in tree_flatten(dl.__all__)[0]
        if inspect.isclass(dl_dict.get(name))
    ]
    all_pairs = [
        (
            f"{dl_dict[c].__module__}.{dl_dict[c].__qualname__}",
            get_parent_depth(dl_dict[c]),
        )
        for c in classes
    ]

    if args.include:
        include_set = set(args.include)
        pairs = [pair for pair in all_pairs if pair[0] in include_set]
        missing = sorted(include_set - {path for path, _ in pairs})
        for path in missing:
            print(f"Warning: requested class not found in dLux exports: {path}")
    else:
        pairs = all_pairs

    if args.limit is not None:
        pairs = pairs[: max(args.limit, 0)]

    UML_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failed = 0

    # Generate UMLs for each class
    for path, depth in pairs:
        print(f"Generating UML for {path}")

        ancestor_levels = max(depth - 3, 0)

        # Generate UML
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pylint.pyreverse.main",
                    "-c",
                    f"{path}",
                    "--output",
                    "html",
                    "-s0",
                    f"-a{ancestor_levels}",
                    "--colorized",
                    "dLux",
                    "--output-directory",
                    str(UML_DIR),
                ],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=PYREVERSE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(f"Warning: pyreverse timed out for {path}, skipping.")
            failed += 1
            continue

        if result.returncode != 0:
            print(f"Warning: pyreverse failed for {path}:")
            if result.stderr.strip():
                print(result.stderr.strip())
            failed += 1
            continue

        # Save to png
        file_name = path.split(".")[-1]
        html_path = UML_DIR / f"{path}.html"
        if not html_path.exists():
            print(
                f"Warning: pyreverse did not generate output for {path}, " "skipping."
            )
            skipped += 1
            continue

        try:
            save_to_png(str(html_path), str(UML_DIR / f"{file_name}.png"))
            generated += 1
        except (WebDriverException, OSError, ValueError) as exc:
            print(f"Warning: screenshot conversion failed for {path}: {exc}")
            failed += 1
        finally:
            # Remove html file if present
            if html_path.exists():
                os.remove(html_path)

    print(
        "UML generation complete: "
        f"generated={generated}, skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
