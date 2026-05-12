# Generated with ChatGPT 5.5
"""
weasy_print.py

Block-based PDF reports using:
    - pandas for tables
    - matplotlib for plots
    - Jinja2 for HTML templating
    - WeasyPrint for HTML/CSS -> PDF

Install:
    pip install pandas matplotlib jinja2 weasyprint

Basic usage:
    import weasy_print as wp

    blocks = [
        wp.header("Sales Report"),
        wp.text("This report summarizes monthly sales."),
        wp.table(df, name="summary_table"),
        wp.page_break(),
        wp.header("Sales Chart", level=2),
        wp.plot(fig, name="sales_chart"),
    ]

    wp.build_pdf(
        blocks,
        "report.pdf",
        title="Sales Report",
        export_assets=True,   # optional; default is False
    )

When export_assets=True, standalone assets are saved to:
    <report_stem>_figueres/

For example:
    report.pdf
    report_figueres/
        sales_chart.pdf
        summary_table.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4
import html
import re
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, BaseLoader, select_autoescape
from weasyprint import HTML


BlockType = Literal[
    "header",
    "text",
    "html",
    "table",
    "plot",
    "image",
    "spacer",
    "page_break",
    "group",
]


@dataclass
class Block:
    type: BlockType
    content: Any = None
    options: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Block helpers
# -----------------------------------------------------------------------------


def header(
    value: str,
    level: int = 1,
    *,
    new_page: bool = False,
    page_break_after: bool = False,
    keep_with_next: bool = False,
    avoid_break_inside: bool = True,
) -> Block:
    return Block(
        "header",
        value,
        {
            "level": level,
            "new_page": new_page,
            "page_break_after": page_break_after,
            "keep_with_next": keep_with_next,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def text(
    value: str,
    *,
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = False,
) -> Block:
    return Block(
        "text",
        value,
        {
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def raw_html(
    value: str,
    *,
    name: str | None = None,
    export: bool = False,
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = False,
) -> Block:
    """Insert trusted raw HTML.

    Do not use this with untrusted input unless you sanitize it first.

    If export_assets=True in build_pdf and export=True here, the raw HTML block
    is also saved as a standalone PDF in the export folder.
    """
    return Block(
        "html",
        value,
        {
            "name": name,
            "export": export,
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def table(
    dataframe: pd.DataFrame,
    *,
    name: str | None = None,
    max_rows: int | None = 30,
    include_index: bool = False,
    classes: str = "data-table",
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = False,
) -> Block:
    return Block(
        "table",
        dataframe,
        {
            "name": name,
            "max_rows": max_rows,
            "include_index": include_index,
            "classes": classes,
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def plot(
    figure: Any,
    *,
    name: str | None = None,
    alt: str = "Chart",
    width: str = "100%",
    dpi: int = 180,
    close: bool = True,
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = True,
) -> Block:
    return Block(
        "plot",
        figure,
        {
            "name": name,
            "alt": alt,
            "width": width,
            "dpi": dpi,
            "close": close,
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def image(
    path: str | Path,
    *,
    name: str | None = None,
    alt: str = "Image",
    width: str = "100%",
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = True,
) -> Block:
    return Block(
        "image",
        path,
        {
            "name": name,
            "alt": alt,
            "width": width,
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def spacer(height: str = "12px") -> Block:
    return Block("spacer", None, {"height": height})


def page_break() -> Block:
    return Block("page_break")


def group(
    *blocks: Block,
    new_page: bool = False,
    page_break_after: bool = False,
    avoid_break_inside: bool = True,
) -> Block:
    """Group blocks inside a div.

    avoid_break_inside=True asks WeasyPrint not to split the group across pages.
    Very large groups can still be split if they cannot fit on a page.
    """
    return Block(
        "group",
        list(blocks),
        {
            "new_page": new_page,
            "page_break_after": page_break_after,
            "avoid_break_inside": avoid_break_inside,
        },
    )


def section(title: str, *blocks: Block, level: int = 1, new_page: bool = True) -> Block:
    """Convenience helper for a titled section."""
    return group(
        header(title, level=level),
        *blocks,
        new_page=new_page,
        avoid_break_inside=False,
    )


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------


class WeasyReport:
    def __init__(
        self,
        output_path: str | Path,
        *,
        title: str = "Report",
        assets_dir: str | Path | None = None,
        keep_assets: bool = False,
        page_size: str = "A4",
        margin: str = "18mm",
        header_text: str | None = None,
        footer: bool = True,
        extra_css: str = "",
        export_assets: bool = False,
        export_dir: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.title = title
        self.keep_assets = keep_assets
        self.page_size = page_size
        self.margin = margin
        self.header_text = header_text
        self.footer = footer
        self.extra_css = extra_css

        # Temporary assets for the main PDF, such as PNG versions of plots.
        # These are deleted by default after the report is built.
        if assets_dir is None:
            self.assets_dir = self.output_path.with_suffix("").parent / f"{self.output_path.stem}_assets"
        else:
            self.assets_dir = Path(assets_dir)

        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Optional thesis/import assets. These are intentionally separate from
        # the temporary report assets and are not deleted after build().
        self.export_assets = export_assets
        if export_dir is None:
            self.export_dir = self.output_path.with_suffix("").parent / f"{self.output_path.stem}_figueres"
        else:
            self.export_dir = Path(export_dir)

        if self.export_assets:
            self.export_dir.mkdir(parents=True, exist_ok=True)

        self._name_counts: dict[str, int] = {}
        self.exported_assets: list[Path] = []

    def build(self, blocks: Sequence[Block | Mapping[str, Any]]) -> list[Path]:
        normalized_blocks = [self._normalize_block(block) for block in blocks]
        rendered_blocks = [self._render_block(block) for block in normalized_blocks]

        html_doc = self._render_template(rendered_blocks)

        # base_url is important: it lets WeasyPrint resolve local image paths.
        HTML(string=html_doc, base_url=str(self.assets_dir.resolve())).write_pdf(str(self.output_path))

        if not self.keep_assets:
            shutil.rmtree(self.assets_dir, ignore_errors=True)

        return self.exported_assets

    # ------------------------------------------------------------------
    # Block rendering
    # ------------------------------------------------------------------

    def _render_block(self, block: Block) -> dict[str, Any]:
        classes = ["block", f"block-{block.type}"]

        if block.options.get("new_page"):
            classes.append("new-page")

        if block.options.get("page_break_after"):
            classes.append("page-break-after")

        if block.options.get("avoid_break_inside"):
            classes.append("avoid-break")

        if block.options.get("keep_with_next"):
            classes.append("keep-with-next")

        if block.type == "header":
            level = int(block.options.get("level", 1))
            level = min(max(level, 1), 6)
            body = f"<h{level}>{html.escape(str(block.content))}</h{level}>"

        elif block.type == "text":
            body = f"<p>{html.escape(str(block.content))}</p>"

        elif block.type == "html":
            body = str(block.content)
            if self.export_assets and block.options.get("export", False):
                self._export_html_pdf(
                    body,
                    name=block.options.get("name"),
                    kind="html",
                    title=block.options.get("name") or "HTML Block",
                )

        elif block.type == "table":
            body = self._render_table(block)
            if self.export_assets:
                self._export_html_pdf(
                    body,
                    name=block.options.get("name"),
                    kind="table",
                    title=block.options.get("name") or "Table",
                )

        elif block.type == "plot":
            body = self._render_plot(block)

        elif block.type == "image":
            body = self._render_image(block)

        elif block.type == "spacer":
            height = html.escape(str(block.options.get("height", "12px")))
            body = f'<div style="height: {height};"></div>'

        elif block.type == "page_break":
            body = ""
            classes.append("manual-page-break")

        elif block.type == "group":
            inner_blocks = [self._normalize_block(b) for b in block.content]
            rendered = [self._render_block(b)["html"] for b in inner_blocks]
            body = "\n".join(rendered)

        else:
            raise ValueError(f"Unknown block type: {block.type}")

        class_attr = " ".join(classes)
        html_block = f'<div class="{class_attr}">\n{body}\n</div>'

        return {
            "classes": class_attr,
            "html": html_block,
        }

    def _render_table(self, block: Block) -> str:
        df = block.content
        if not isinstance(df, pd.DataFrame):
            raise TypeError("table() expects a pandas DataFrame")

        max_rows = block.options.get("max_rows", 30)
        include_index = bool(block.options.get("include_index", False))
        classes = str(block.options.get("classes", "data-table"))

        table_df = df.copy()
        if max_rows is not None:
            table_df = table_df.head(int(max_rows))

        return table_df.to_html(
            index=include_index,
            classes=classes,
            border=0,
            escape=True,
        )

    def _render_plot(self, block: Block) -> str:
        fig = block.content

        if self.export_assets:
            export_name = block.options.get("name") or block.options.get("alt") or "plot"
            export_path = self._next_export_path(kind="plot", name=export_name, suffix=".pdf")
            fig.savefig(export_path, format="pdf", bbox_inches="tight")
            self.exported_assets.append(export_path)

        # PNG version for embedding in the main HTML report.
        filename = f"plot_{uuid4().hex}.png"
        path = self.assets_dir / filename

        dpi = int(block.options.get("dpi", 180))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

        if block.options.get("close", True):
            plt.close(fig)

        alt = html.escape(str(block.options.get("alt", "Chart")))
        width = html.escape(str(block.options.get("width", "100%")))
        return f'<img src="{filename}" alt="{alt}" style="width: {width};" />'

    def _render_image(self, block: Block) -> str:
        src_path = Path(block.content)
        if not src_path.exists():
            raise FileNotFoundError(src_path)

        if self.export_assets:
            export_name = block.options.get("name") or src_path.stem or block.options.get("alt") or "image"
            export_path = self._next_export_path(kind="image", name=export_name, suffix=".pdf")
            self._export_image_pdf(src_path, export_path, alt=str(block.options.get("alt", "Image")))
            self.exported_assets.append(export_path)

        # Copy image into temporary assets folder for the main report.
        filename = f"image_{uuid4().hex}{src_path.suffix}"
        dest = self.assets_dir / filename
        shutil.copy2(src_path, dest)

        alt = html.escape(str(block.options.get("alt", "Image")))
        width = html.escape(str(block.options.get("width", "100%")))
        return f'<img src="{filename}" alt="{alt}" style="width: {width};" />'

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _next_export_path(self, *, kind: str, name: str | None, suffix: str) -> Path:
        base_name = self._safe_filename(name or kind)

        if not base_name or base_name == "untitled":
            base_name = kind

        key = base_name
        count = self._name_counts.get(key, 0) + 1
        self._name_counts[key] = count

        if count == 1:
            filename = f"{base_name}{suffix}"
        else:
            filename = f"{base_name}_{count:02d}{suffix}"

        return self.export_dir / filename

    @staticmethod
    def _safe_filename(value: str) -> str:
        value = str(value).strip().lower()
        value = re.sub(r"[^a-z0-9._ -]+", "", value)
        value = re.sub(r"[\s-]+", "_", value)
        value = value.strip("._")
        return value[:100] or "untitled"

    def _export_html_pdf(self, body_html: str, *, name: str | None, kind: str, title: str) -> None:
        export_path = self._next_export_path(kind=kind, name=name, suffix=".pdf")

        standalone_html = self._render_standalone_template(
            body_html=body_html,
            title=title,
            css=self._css(export_mode=True),
        )

        HTML(string=standalone_html, base_url=str(self.assets_dir.resolve())).write_pdf(str(export_path))
        self.exported_assets.append(export_path)

    def _export_image_pdf(self, src_path: Path, export_path: Path, *, alt: str = "Image") -> None:
        if src_path.suffix.lower() == ".pdf":
            shutil.copy2(src_path, export_path)
            return

        img_uri = src_path.resolve().as_uri()
        safe_alt = html.escape(alt)
        body_html = f'<img src="{img_uri}" alt="{safe_alt}" style="max-width: 100%; height: auto;" />'

        standalone_html = self._render_standalone_template(
            body_html=body_html,
            title=alt,
            css=self._css(export_mode=True),
        )

        HTML(string=standalone_html, base_url=str(src_path.parent.resolve())).write_pdf(str(export_path))

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_template(self, rendered_blocks: list[dict[str, Any]]) -> str:
        env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.from_string(DEFAULT_TEMPLATE)

        return template.render(
            title=self.title,
            blocks=rendered_blocks,
            css=self._css(),
        )

    def _render_standalone_template(self, *, body_html: str, title: str, css: str) -> str:
        env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.from_string(STANDALONE_TEMPLATE)

        return template.render(
            title=title,
            body_html=body_html,
            css=css,
        )

    def _css(self, *, export_mode: bool = False) -> str:
        header_css = ""
        if self.header_text and not export_mode:
            safe_header = self.header_text.replace('"', '\\"')
            header_css = f'''
            @top-center {{
                content: "{safe_header}";
                font-size: 9pt;
                color: #666;
            }}
            '''

        footer_css = ""
        if self.footer and not export_mode:
            footer_css = '''
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }
            '''

        export_page_size = "A4" if export_mode else self.page_size
        export_margin = "8mm" if export_mode else self.margin

        return f'''
        @page {{
            size: {export_page_size};
            margin: {export_margin};
            {header_css}
            {footer_css}
        }}

        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 10pt;
            line-height: 1.45;
            color: #222;
        }}

        h1 {{
            font-size: 22pt;
            margin: 0 0 16px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid #ddd;
        }}

        h2 {{
            font-size: 16pt;
            margin: 18px 0 10px 0;
        }}

        h3 {{
            font-size: 12pt;
            margin: 14px 0 8px 0;
        }}

        p {{
            margin: 0 0 12px 0;
        }}

        img {{
            display: block;
            max-width: 100%;
            margin: 8px 0 16px 0;
        }}

        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0 18px 0;
            font-size: 8pt;
        }}

        table.data-table th {{
            background: #eeeeee;
            font-weight: bold;
        }}

        table.data-table th,
        table.data-table td {{
            border: 1px solid #cccccc;
            padding: 4px 6px;
            vertical-align: top;
        }}

        table.data-table tr:nth-child(even) td {{
            background: #fafafa;
        }}

        .new-page {{
            break-before: page;
            page-break-before: always;
        }}

        .page-break-after {{
            break-after: page;
            page-break-after: always;
        }}

        .manual-page-break {{
            break-before: page;
            page-break-before: always;
            height: 0;
        }}

        .avoid-break {{
            break-inside: avoid;
            page-break-inside: avoid;
        }}

        .keep-with-next {{
            break-after: avoid;
            page-break-after: avoid;
        }}

        .block {{
            margin-bottom: 10px;
        }}

        {self.extra_css}
        '''

    def _normalize_block(self, raw_block: Block | Mapping[str, Any]) -> Block:
        if isinstance(raw_block, Block):
            return raw_block

        if isinstance(raw_block, Mapping):
            block_type = raw_block.get("type")
            if not block_type:
                raise ValueError("Block dictionaries must contain a 'type' key")

            content = raw_block.get("content", raw_block.get("data", raw_block.get("text")))
            options = {k: v for k, v in raw_block.items() if k not in {"type", "content", "data", "text"}}
            return Block(block_type, content, options)  # type: ignore[arg-type]

        raise TypeError(f"Expected Block or dict, got {type(raw_block)!r}")


DEFAULT_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>{{ css | safe }}</style>
</head>
<body>
    {% for block in blocks %}
        {{ block.html | safe }}
    {% endfor %}
</body>
</html>
"""


STANDALONE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>{{ css | safe }}</style>
</head>
<body>
    {{ body_html | safe }}
</body>
</html>
"""


def build_pdf(
    blocks: Sequence[Block | Mapping[str, Any]],
    output_path: str | Path,
    **options: Any,
) -> list[Path]:
    report = WeasyReport(output_path, **options)
    return report.build(blocks)


__all__ = [
    "Block",
    "WeasyReport",
    "build_pdf",
    "header",
    "text",
    "raw_html",
    "table",
    "plot",
    "image",
    "spacer",
    "page_break",
    "group",
    "section",
]
