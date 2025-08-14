"""
Markdown to HTML conversion utility.

Usage (CLI):
    python utils/markdown_2_html.py outputs/YourReport.md

Function:
    convert_markdown_file_to_html(path) -> str  # returns output HTML path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final
from typing import Dict

import markdown


HTML_TEMPLATE: Final[str] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <style>
        :root {{
            --background-color: #f5f5f7;
            --text-color: #1d1d1f;
            --heading-blue: #0071e3;
            --link-color: #0066cc;
            --border-color: #d2d2d7;
            --card-bg: #ffffff;
        }}

        html, body {{
            margin: 0;
            padding: 0;
            background: var(--background-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
        }}

        a {{ color: var(--link-color); }}

        .container {{
            max-width: 980px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        .content {{
            background: var(--card-bg);
            border-radius: 18px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
            padding: 40px;
        }}

        h1 {{
            color: var(--heading-blue);
            font-size: 40px;
            font-weight: 700;
            letter-spacing: -0.022em;
            margin: 0 0 0.75em 0;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #e8f3ff;
        }}
        h2 {{
            color: var(--heading-blue);
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.022em;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #e8f3ff;
        }}
        h3 {{ font-size: 24px; font-weight: 600; margin-top: 1.25em; }}
        h4 {{ font-size: 20px; font-weight: 600; margin-top: 1.25em; }}

        p {{ margin: 0 0 1em 0; }}
        ul, ol {{ margin: 0 0 1em 1.5em; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 0.95rem;
        }}
        th, td {{
            border: 1px solid var(--border-color);
            padding: 8px 10px;
            text-align: left;
        }}
        thead th {{
            background: #f0f7ff;
        }}
        tbody tr:nth-child(even) {{
            background: #fafafa;
        }}

        pre {{
            background: #0b1021;
            color: #e6e6e6;
            padding: 12px 14px;
            overflow: auto;
            border-radius: 8px;
        }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 4px;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 20px; }}
            .content {{ padding: 20px; }}
            h1 {{ font-size: 32px; }}
            h2 {{ font-size: 26px; }}
        }}
    </style>
    <!-- Plotly inline HTML will be preserved by the Markdown renderer below -->
    <meta http-equiv="Content-Security-Policy" content="script-src 'self' 'unsafe-inline' 'unsafe-eval' data: https://cdn.plot.ly; object-src 'self' data:;" />
</head>
<body>
    <div class="container">
        <div class="content">
            {content}
        </div>
    </div>
</body>
</html>
"""


def _render_markdown(markdown_text: str) -> str:
    """Render Markdown text to HTML body.

    - Enables tables and fenced code blocks
    - Preserves raw HTML blocks (e.g., Plotly charts embedded in the Markdown)
    """

    md = markdown.Markdown(
        extensions=[
            "extra",        # includes sensible defaults; supports tables, etc.
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
            "attr_list",
        ],
        output_format="html5",
    )
    return md.convert(markdown_text)


#def convert_markdown_file_to_html(markdown_file_path: str) -> str:
def convert_markdown_file_to_html(station_code: str, target_date: str) -> Dict[str, str]:
    """Convert a Markdown file to an HTML file for a given station and target date.

    Args:
        station_code: Station code (such as "DNA6)
        target_date: Target date in YYYY-MM-DD format (such as "2025-08-14")

    Returns:
        {
            "status": "success" | "error",
            "html_file_path": "The absolute path to the generated .html file.",
            "error_message": "The error message if the status is error"
        }
    """
    markdown_file_path = f"/Users/hungjen/Documents/Projects/COHDD/outputs/COHDD-{station_code}-{target_date}.md"
    try:    
        source_path = Path(markdown_file_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
           return {
                "status": "error",
                "error_message": f"Markdown file not found: {source_path}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

    if source_path.suffix.lower() not in {".md", ".markdown"}:
        #raise ValueError("Input file must have a .md or .markdown extension")
        return {
            "status": "error",
            "error_message": f"Input file must have a .md or .markdown extension: {source_path}"
        }

    try:
        markdown_text = source_path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error reading markdown file: {e}"
        }

    # Render body HTML and wrap with template
    try:
        body_html = _render_markdown(markdown_text)
        title = source_path.stem
        full_html = HTML_TEMPLATE.format(title=title, content=body_html)
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error rendering markdown file: {e}"
        }

    output_path = source_path.with_suffix(".html")
    output_path.write_text(full_html, encoding="utf-8")

    return {
        "status": "success",
        "html_file_path": str(output_path)
    }


# modify this CLI in next version
def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python utils/markdown_2_html.py outputs/YourFile.md")
        return 2

    input_path = argv[1]
    try:
        output_path = convert_markdown_file_to_html(input_path)
    except Exception as exc:  # noqa: BLE001 - surface clear error to user
        print(f"Error: {exc}")
        return 1

    print(output_path)
    return 0


#if __name__ == "__main__":
#    raise SystemExit(_main(sys.argv))