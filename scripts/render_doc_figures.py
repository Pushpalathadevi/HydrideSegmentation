"""Generate publication-style SVG figures for the documentation site.

The repository prefers static SVG for schematics and flow sheets so the docs
stay crisp, offline-friendly, and easy to inspect in HTML or PDF builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DIAGRAMS = DOCS / "diagrams"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    text: str
    fill: str = "#eef5fb"
    stroke: str = "#5b7a99"
    text_fill: str = "#10263f"
    size: int = 20
    weight: int = 600
    radius: int = 16


def _tspan_lines(text: str, cx: float, cy: float, size: int, color: str, weight: int) -> str:
    lines = text.split("\n")
    line_gap = size * 1.2
    first_y = cy - ((len(lines) - 1) * line_gap) / 2
    parts = [
        f'<text x="{cx}" y="{first_y:.1f}" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        parts.append(f'<tspan x="{cx}" dy="{dy:.1f}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _box(box: Box) -> str:
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    return (
        f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="{box.radius}" ry="{box.radius}" '
        f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2"/>'
        + _tspan_lines(box.text, cx, cy, box.size, box.text_fill, box.weight)
    )


def _panel(x: float, y: float, w: float, h: float, title: str, fill: str = "#f8fbff", stroke: str = "#a9bfd4") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" ry="22" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        + f'<text x="{x + 24}" y="{y + 34}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="20" font-weight="700" fill="#10263f">{escape(title)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#5a7388", width: int = 3) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'
    )


def _poly(points: list[tuple[float, float]], fill: str, stroke: str, text: str, x: float, y: float, w: float, h: float) -> str:
    pts = " ".join(f"{px},{py}" for px, py in points)
    cx = x + w / 2
    cy = y + h / 2
    return (
        f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        + _tspan_lines(text, cx, cy, 18, "#10263f", 700)
    )


def _header(title: str, subtitle: str | None = None, width: int = 1600, height: int = 0) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="100%" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#5a7388"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="24" font-weight="700" fill="#10263f">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width/2}" y="60" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="14" fill="#4b6075">{escape(subtitle)}</text>'
        )
    return "".join(parts)


def _footer() -> str:
    return "</svg>"


def svg_learning_path() -> str:
    width, height = 1600, 760
    parts = [_header("Beginner Learning Path", "Read first on the left, then do and review on the right.", width, height)]
    parts.append(_panel(70, 92, 1460, 620, "Recommended flow"))
    parts.append(_box(Box(540, 128, 520, 74, "Mission + principles", fill="#f6ecd7", stroke="#b28b3e", size=24)))
    parts.append(_line(800, 202, 800, 244))
    parts.append(_panel(110, 250, 610, 330, "Read first", fill="#fbfdff"))
    parts.append(_panel(880, 250, 610, 330, "Do and review", fill="#fbfdff"))

    left = [
        Box(170, 300, 450, 58, "Glossary + learning path", fill="#e8f0fb"),
        Box(170, 390, 450, 62, "Notebook 01\nData preparation", fill="#eef5fb"),
        Box(170, 485, 450, 62, "Notebook 02\nTraining + inference", fill="#eef5fb"),
    ]
    right = [
        Box(940, 300, 450, 62, "GUI\nInspect, compare, correct", fill="#f7efd8"),
        Box(940, 390, 450, 62, "Notebook 03\nCorrection + export", fill="#eef5fb"),
        Box(940, 485, 450, 62, "Notebook 04\nEvaluation + reports", fill="#eef5fb"),
    ]
    for b in left + right:
        parts.append(_box(b))
    parts.append(_box(Box(540, 580, 520, 64, "Contribute new experiments", fill="#eaf6ea", stroke="#5d8d5d", size=21)))
    parts.append(_line(800, 560, 800, 580))
    parts.append(_line(620, 356, 620, 417))
    parts.append(_line(620, 455, 620, 485))
    parts.append(_line(1380, 356, 1380, 421))
    parts.append(_line(1380, 453, 1380, 517))
    parts.append(_line(620, 516, 540, 612))
    parts.append(_line(1380, 548, 1060, 612))
    parts.append(_line(800, 646, 800, 706))
    parts.append(_box(Box(500, 692, 600, 48, "Start with sample data and the notebook ladder before larger datasets.", fill="#f8fbff", stroke="#a9bfd4", size=16, weight=500)))
    parts.append(_footer())
    return "".join(parts)


def svg_worked_example() -> str:
    width, height = 1600, 640
    parts = [_header("Worked Example: Conventional Baseline vs ML Model", "Side-by-side comparison uses the same image and the same review criteria.", width, height)]
    parts.append(_panel(70, 92, 1460, 500, "Comparison workflow"))
    parts.append(_box(Box(590, 130, 420, 64, "Same input image", fill="#f6ecd7", stroke="#b28b3e", size=24)))
    parts.append(_line(800, 194, 800, 230))
    parts.append(_panel(110, 240, 630, 260, "Conventional", fill="#fbfdff"))
    parts.append(_panel(860, 240, 630, 260, "ML model", fill="#fbfdff"))
    left = [
        Box(190, 290, 470, 56, "Baseline", fill="#e8f0fb"),
        Box(190, 370, 470, 56, "Mask + overlay", fill="#eef5fb"),
        Box(190, 450, 470, 56, "Rule-based interpretation", fill="#eef5fb"),
    ]
    right = [
        Box(940, 290, 470, 56, "Trained model", fill="#f7efd8"),
        Box(940, 370, 470, 56, "Mask + overlay", fill="#eef5fb"),
        Box(940, 450, 470, 56, "Learned interpretation", fill="#eef5fb"),
    ]
    for b in left + right:
        parts.append(_box(b))
    parts.append(_box(Box(500, 542, 600, 56, "Compare boundaries, counts, morphology, and failure modes", fill="#eaf6ea", stroke="#5d8d5d", size=18)))
    parts.append(_box(Box(520, 610, 560, 40, "Read metrics together with the images", fill="#ffffff", stroke="#d8dfe8", size=15, weight=500)))
    parts.append(_line(800, 484, 800, 542))
    parts.append(_line(660, 346, 660, 370))
    parts.append(_line(660, 426, 660, 450))
    parts.append(_line(1340, 346, 1340, 370))
    parts.append(_line(1340, 426, 1340, 450))
    parts.append(_footer())
    return "".join(parts)


def svg_model_selection() -> str:
    width, height = 1600, 760
    parts = [_header("Which Model Should I Use?", "Start with the simplest model that can reasonably solve the problem.", width, height)]
    parts.append(_panel(70, 92, 1460, 620, "Selection tree"))
    parts.append(_poly([(702, 140), (898, 140), (960, 202), (800, 268), (640, 202)], "#f6ecd7", "#b28b3e", "Need a quick\nbaseline?", 640, 140, 320, 128))
    parts.append(_box(Box(140, 320, 430, 68, "Use conventional segmentation", fill="#e8f0fb", size=23)))
    parts.append(_box(Box(1030, 320, 430, 68, "Need a trainable model?", fill="#f7efd8", size=23)))
    parts.append(_line(640, 204, 570, 354))
    parts.append(_line(960, 204, 1030, 354))
    parts.append(_box(Box(960, 410, 500, 56, "Dataset small or medium?", fill="#eef5fb", size=20)))
    parts.append(_box(Box(960, 492, 500, 56, "Prefer unet_binary or smp_unet_resnet18", fill="#eef5fb", size=18)))
    parts.append(_box(Box(960, 574, 500, 56, "Need more global context?", fill="#eef5fb", size=20)))
    parts.append(_box(Box(960, 656, 500, 56, "Try hf_segformer_b0 or hf_segformer_b2", fill="#eef5fb", size=18)))
    parts.append(_box(Box(140, 410, 430, 56, "Use the conventional baseline first", fill="#eef5fb", size=18)))
    parts.append(_line(1220, 388, 1220, 410))
    parts.append(_line(1220, 466, 1220, 492))
    parts.append(_line(1220, 548, 1220, 574))
    parts.append(_line(1220, 630, 1220, 656))
    parts.append(_box(Box(140, 520, 430, 58, "If not, keep the baseline and document why.", fill="#ffffff", stroke="#d8dfe8", size=16, weight=500)))
    parts.append(_box(Box(540, 520, 300, 58, "Use the compact transformer first", fill="#eef5fb", size=17)))
    parts.append(_footer())
    return "".join(parts)


def svg_conventional_pipeline() -> str:
    width, height = 1600, 640
    parts = [_header("Conventional Segmentation Pipeline", "CPU-first flow sheet with the same scientific stages as the runtime baseline.", width, height)]
    parts.append(_panel(70, 92, 1460, 500, "Flow sheet"))
    x_positions = [110, 285, 460, 635, 810, 985, 1160, 1335]
    boxes = [
        "Input image\nRGB or grayscale",
        "Load grayscale\nview",
        "Crop bottom\nregion?",
        "CLAHE contrast\nnormalization",
        "Gaussian blur /\nlocal smoothing",
        "Adaptive local\nthreshold",
        "Morphological\nclosing",
        "Connected-component\nlabeling",
    ]
    fills = ["#f6ecd7", "#eef5fb", "#fff3cf", "#eef5fb", "#eef5fb", "#eef5fb", "#eef5fb", "#eef5fb"]
    widths = [140, 145, 145, 150, 145, 150, 145, 165]
    for x, text, fill, w in zip(x_positions, boxes, fills, widths):
        parts.append(_box(Box(x, 260, w, 84, text, fill=fill, size=17)))
    for x1, x2 in zip([250, 425, 600, 785, 960, 1135, 1310], [285, 460, 635, 810, 985, 1160, 1335]):
        parts.append(_line(x1, 302, x2, 302))
    parts.append(_poly([(660, 376), (720, 340), (780, 376), (720, 412)], "#fdf4d9", "#b28b3e", "Crop\nbottom?", 660, 340, 120, 72))
    parts.append(_line(720, 344, 720, 260))
    parts.append(_line(720, 412, 720, 260))
    parts.append(_box(Box(605, 440, 230, 64, "Area-based filtering", fill="#eaf6ea", stroke="#5d8d5d", size=19)))
    parts.append(_box(Box(855, 440, 220, 64, "Binary mask", fill="#e8f0fb", size=20)))
    parts.append(_box(Box(1095, 440, 250, 64, "Optional overlay /\ncontour preview", fill="#eef5fb", size=18)))
    parts.append(_box(Box(1365, 440, 180, 64, "ORA export\nand/or PNG mask", fill="#eef5fb", size=18)))
    parts.append(_line(1395, 344, 1395, 440))
    parts.append(_line(720, 412, 720, 440))
    parts.append(_line(835, 472, 855, 472))
    parts.append(_line(1075, 472, 1095, 472))
    parts.append(_line(1345, 472, 1365, 472))
    parts.append(_footer())
    return "".join(parts)


def svg_gui_model_integration() -> str:
    width, height = 1600, 620
    parts = [_header("GUI Model Integration Guide", "From checkpoint validation to a GUI inference smoke test.", width, height)]
    parts.append(_panel(70, 92, 1460, 460, "Integration workflow"))
    xs = [110, 300, 490, 680, 870, 1060, 1250]
    texts = [
        "Train or obtain\na checkpoint",
        "Validate the\ncheckpoint or bundle",
        "Register the model\nin the frozen registry",
        "Expose the model in\nGUI metadata",
        "Confirm the backend loader\ncan resolve it",
        "Run a GUI inference\nsmoke test",
        "Inspect output and\ndocument the run",
    ]
    fills = ["#f6ecd7", "#eef5fb", "#eef5fb", "#eef5fb", "#eef5fb", "#eaf6ea", "#eef5fb"]
    widths = [165, 175, 175, 165, 175, 165, 165]
    for x, text, fill, w in zip(xs, texts, fills, widths):
        parts.append(_box(Box(x, 250, w, 90, text, fill=fill, size=17)))
    for x1, x2 in zip([275, 465, 655, 835, 1025, 1215], [300, 490, 680, 870, 1060, 1250]):
        parts.append(_line(x1, 295, x2, 295))
    parts.append(_box(Box(635, 380, 330, 58, "The GUI label is for users; the model ID is for runtime.", fill="#ffffff", stroke="#d8dfe8", size=16, weight=500)))
    parts.append(_footer())
    return "".join(parts)


def svg_workflow_roadmap() -> str:
    width, height = 1600, 740
    parts = [_header("Deployment And Productization Roadmap", "Build-and-gate on the left, deploy-and-learn on the right.", width, height)]
    parts.append(_panel(70, 92, 1460, 560, "Target operating architecture"))
    parts.append(_box(Box(510, 126, 580, 66, "Dataset contract\n(train / val / test + manifest)", fill="#f6ecd7", stroke="#b28b3e", size=20)))
    parts.append(_line(800, 192, 800, 236))
    parts.append(_panel(130, 240, 610, 320, "Build and gate"))
    parts.append(_panel(860, 240, 610, 320, "Deploy and learn"))
    left = [
        Box(180, 292, 500, 54, "Train / eval orchestration", fill="#eef5fb", size=18),
        Box(180, 370, 500, 54, "Benchmark artifacts", fill="#eef5fb", size=18),
        Box(180, 448, 500, 54, "Promotion gate", fill="#eef5fb", size=18),
    ]
    right = [
        Box(930, 292, 500, 54, "Deployment package", fill="#eef5fb", size=18),
        Box(930, 370, 500, 54, "Runtime inference app", fill="#eef5fb", size=18),
        Box(930, 448, 500, 54, "Ops feedback", fill="#eef5fb", size=18),
    ]
    for b in left + right:
        parts.append(_box(b))
    parts.append(_line(430, 346, 430, 370))
    parts.append(_line(430, 424, 430, 448))
    parts.append(_line(430, 502, 800, 502))
    parts.append(_line(1060, 346, 1060, 370))
    parts.append(_line(1060, 424, 1060, 448))
    parts.append(_line(1060, 502, 800, 502))
    parts.append(_box(Box(520, 558, 560, 52, "Quality, robustness, runtime, and reproducibility are the promotion gates.", fill="#eaf6ea", stroke="#5d8d5d", size=16, weight=500)))
    parts.append(_footer())
    return "".join(parts)


def svg_model_architecture() -> str:
    width, height = 1680, 1700
    parts = [_header("Model Architecture Manuscript Foundation", "A stacked SVG keeps the family overview and the backend-specific flows compact.", width, height)]
    # Panel 1 overview
    parts.append(_panel(70, 86, 1540, 340, "Architecture overview"))
    parts.append(_box(Box(650, 130, 380, 60, "Input image", fill="#f6ecd7", stroke="#b28b3e", size=22)))
    parts.append(_line(840, 190, 840, 226))
    parts.append(_panel(110, 230, 1460, 160, "Model families"))
    columns = [
        (170, "CNN family", ["unet_binary", "smp_unet_resnet18", "smp_deeplabv3plus_resnet101", "smp_unetplusplus_resnet101", "smp_pspnet_resnet101", "smp_fpn_resnet101"], "#eef5fb"),
        (615, "Transformer family", ["hf_segformer_b0", "hf_segformer_b2", "hf_segformer_b5", "hf_upernet_swin_large"], "#f7efd8"),
        (1080, "Hybrid variants", ["transunet_tiny", "segformer_mini"], "#eaf6ea"),
    ]
    for x, title, items, fill in columns:
        parts.append(_box(Box(x, 276, 330 if len(items) > 2 else 270, 26, title, fill=fill, stroke="#8aa4bf", size=14, weight=700, radius=10)))
        row_y = 312
        for item in items:
            w = 330 if len(item) > 16 else 270
            parts.append(_box(Box(x, row_y, w, 26, item, fill="#ffffff", stroke="#c9d6e4", size=13, weight=500, radius=10)))
            row_y += 28
    parts.append(_box(Box(665, 368, 350, 34, "Use the family overview as a manuscript figure legend.", fill="#ffffff", stroke="#d8dfe8", size=12, weight=500, radius=10)))

    # Panel 2 U-Net style
    y0 = 450
    parts.append(_panel(70, y0, 1540, 270, "U-Net-style decoder flow"))
    parts.append(_box(Box(130, y0 + 58, 180, 54, "Input", fill="#f6ecd7", size=19)))
    parts.append(_box(Box(340, y0 + 58, 170, 54, "Encoder 1", fill="#eef5fb", size=17)))
    parts.append(_box(Box(540, y0 + 58, 170, 54, "Encoder 2", fill="#eef5fb", size=17)))
    parts.append(_box(Box(740, y0 + 58, 170, 54, "Bottleneck", fill="#eef5fb", size=17)))
    parts.append(_box(Box(940, y0 + 58, 170, 54, "Decoder 2", fill="#eef5fb", size=17)))
    parts.append(_box(Box(1140, y0 + 58, 170, 54, "Decoder 1", fill="#eef5fb", size=17)))
    parts.append(_box(Box(1340, y0 + 58, 170, 54, "1x1 head", fill="#eef5fb", size=17)))
    parts.append(_box(Box(1480, y0 + 58, 90, 54, "Logits", fill="#eaf6ea", size=16)))
    for x1, x2 in zip([310, 510, 710, 910, 1110, 1310, 1480], [340, 540, 740, 940, 1140, 1340, 1480]):
        parts.append(_line(x1, y0 + 85, x2, y0 + 85))
    parts.append(_line(600, y0 + 58, 1010, y0 + 112))
    parts.append(_line(400, y0 + 58, 1210, y0 + 112))
    parts.append(_box(Box(530, y0 + 132, 620, 48, "Skip connections preserve spatial detail during decoding.", fill="#ffffff", stroke="#d8dfe8", size=15, weight=500)))

    # Panel 3 SegFormer
    y1 = 760
    parts.append(_panel(70, y1, 1540, 250, "SegFormer-style flow"))
    seg_x = [160, 340, 520, 700, 880, 1080]
    seg_texts = ["Input", "Patch embedding", "Stage 1", "Stage 2", "Stage 3", "MLP head"]
    seg_widths = [120, 160, 110, 110, 110, 120]
    for x, text, w in zip(seg_x, seg_texts, seg_widths):
        parts.append(_box(Box(x, y1 + 66, w, 54, text, fill="#eef5fb" if text != "Input" else "#f6ecd7", size=17)))
    parts.append(_line(280, y1 + 93, 340, y1 + 93))
    parts.append(_line(500, y1 + 93, 520, y1 + 93))
    parts.append(_line(630, y1 + 93, 700, y1 + 93))
    parts.append(_line(810, y1 + 93, 880, y1 + 93))
    parts.append(_line(1000, y1 + 93, 1080, y1 + 93))
    parts.append(_box(Box(1240, y1 + 66, 260, 54, "Lightweight decode head\nkeeps compute focused upstream.", fill="#ffffff", stroke="#d8dfe8", size=14, weight=500)))

    # Panel 4 Hybrid
    y2 = 1060
    parts.append(_panel(70, y2, 1540, 250, "Hybrid flow"))
    hy_x = [150, 390, 640, 900, 1160]
    hy_texts = ["Input", "Convolutional stem", "Transformer block(s)", "Reshape to feature map", "Decoder"]
    hy_widths = [120, 190, 200, 220, 140]
    for x, text, w in zip(hy_x, hy_texts, hy_widths):
        parts.append(_box(Box(x, y2 + 70, w, 54, text, fill="#eef5fb" if text != "Input" else "#f6ecd7", size=17)))
    for x1, x2 in zip([270, 580, 840, 1100], [390, 640, 900, 1160]):
        parts.append(_line(x1, y2 + 97, x2, y2 + 97))
    parts.append(_box(Box(1330, y2 + 70, 120, 54, "Logits", fill="#eaf6ea", size=17)))
    parts.append(_line(1300, y2 + 97, 1330, y2 + 97))
    parts.append(_box(Box(520, y2 + 136, 640, 48, "Bridge models test whether global context improves the baseline.", fill="#ffffff", stroke="#d8dfe8", size=15, weight=500)))

    parts.append(_footer())
    return "".join(parts)


def svg_code_architecture_map() -> str:
    width, height = 1700, 2000
    parts = [_header("Code Architecture and Data Flow Map", "Three stacked views keep the repository map readable without a giant horizontal canvas.", width, height)]

    # Panel 1 system architecture
    y0 = 90
    parts.append(_panel(50, y0, 1600, 580, "System architecture"))
    top_boxes = [
        (110, 160, 250, 64, "Client surfaces\nQt / CLI / Service", "#f6ecd7"),
        (420, 160, 330, 64, "App orchestration\nsrc/microseg/app", "#eef5fb"),
        (820, 160, 320, 64, "Segmentation core\nsrc/microseg", "#eef5fb"),
        (1190, 160, 290, 64, "Data lifecycle\nprep + dataops", "#eef5fb"),
        (110, 280, 290, 64, "Configs + artifacts\nconfigs, outputs, registry", "#eaf6ea"),
        (460, 280, 290, 64, "Training backends\nunet / pixel / HF / SMP", "#f7efd8"),
        (820, 280, 290, 64, "Governance + deployment\nquality + deployment", "#eef5fb"),
        (1180, 280, 320, 64, "Compatibility layer\nlegacy adapters", "#eef5fb"),
    ]
    for x, y, w, h, text, fill in top_boxes:
        parts.append(_box(Box(x, y0 + y, w, h, text, fill=fill, size=16)))
    parts.append(_line(360, y0 + 192, 420, y0 + 192))
    parts.append(_line(750, y0 + 192, 820, y0 + 192))
    parts.append(_line(1140, y0 + 192, 1190, y0 + 192))
    parts.append(_line(260, y0 + 256, 260, y0 + 280))
    parts.append(_line(590, y0 + 256, 590, y0 + 280))
    parts.append(_line(960, y0 + 256, 960, y0 + 280))
    parts.append(_line(1340, y0 + 256, 1340, y0 + 280))
    parts.append(_box(Box(200, y0 + 382, 1300, 120, "The layers remain modular: UI, orchestration, core pipelines, dataops, models, and deployment are separate.", fill="#ffffff", stroke="#d8dfe8", size=18, weight=500)))

    # Panel 2 data flow
    y1 = 730
    parts.append(_panel(50, y1, 1600, 410, "End-to-end data flow"))
    flow = [
        (120, "Raw images\n+ masks", "#f6ecd7"),
        (350, "Dataset\npreparation", "#eef5fb"),
        (580, "Curated dataset\nwith manifests", "#eef5fb"),
        (830, "Training", "#eef5fb"),
        (1030, "Model checkpoints\n+ logs", "#eef5fb"),
        (1240, "Evaluation\n+ analysis", "#eef5fb"),
        (1450, "Reports\nJSON / HTML / CSV", "#eaf6ea"),
    ]
    for x, text, fill in flow:
        parts.append(_box(Box(x, y1 + 110, 160, 72, text, fill=fill, size=16)))
    for x1, x2 in zip([280, 510, 760, 960, 1160, 1370], [350, 580, 830, 1030, 1240, 1450]):
        parts.append(_line(x1, y1 + 146, x2, y1 + 146))
    parts.append(_box(Box(1180, y1 + 255, 430, 60, "Package and promote only after validation passes.", fill="#ffffff", stroke="#d8dfe8", size=16, weight=500)))
    parts.append(_line(1110, y1 + 182, 1180, y1 + 255))
    parts.append(_line(1250, y1 + 182, 1395, y1 + 255))

    # Panel 3 runtime interaction
    y2 = 1190
    parts.append(_panel(50, y2, 1600, 700, "Qt runtime interaction"))
    runtime = [
        (120, "User", "#f6ecd7"),
        (310, "GUI\nmain window", "#eef5fb"),
        (560, "Desktop\nworkflow", "#eef5fb"),
        (810, "Segmentation\npipeline", "#eef5fb"),
        (1060, "Predictor", "#eef5fb"),
        (1260, "Analyzer", "#eef5fb"),
        (1450, "Export\nartifact", "#eaf6ea"),
    ]
    for x, text, fill in runtime:
        parts.append(_box(Box(x, y2 + 150, 160, 72, text, fill=fill, size=16)))
    for x1, x2 in zip([280, 500, 750, 1000, 1200, 1400], [310, 560, 810, 1060, 1260, 1450]):
        parts.append(_line(x1, y2 + 186, x2, y2 + 186))
    parts.append(_box(Box(360, y2 + 290, 980, 78, "The GUI and the core stay decoupled; the UI only receives status, result metadata, and export locations.", fill="#ffffff", stroke="#d8dfe8", size=16, weight=500)))
    parts.append(_line(1460, y2 + 222, 1110, y2 + 290))
    parts.append(_line(1060, y2 + 222, 850, y2 + 290))
    parts.append(_footer())
    return "".join(parts)


def svg_code_architecture_system_view() -> str:
    width, height = 1600, 620
    parts = [_header("Code Architecture and Data Flow Map", "System architecture view", width, height)]
    parts.append(_panel(50, 90, 1500, 480, "System architecture"))
    boxes = [
        (110, 170, 250, 64, "Client surfaces\nQt / CLI / Service", "#f6ecd7"),
        (420, 170, 330, 64, "App orchestration\nsrc/microseg/app", "#eef5fb"),
        (820, 170, 320, 64, "Segmentation core\nsrc/microseg", "#eef5fb"),
        (1190, 170, 280, 64, "Data lifecycle\nprep + dataops", "#eef5fb"),
        (110, 290, 290, 64, "Configs + artifacts", "#eaf6ea"),
        (460, 290, 290, 64, "Training backends", "#f7efd8"),
        (820, 290, 290, 64, "Governance + deployment", "#eef5fb"),
        (1180, 290, 300, 64, "Compatibility layer", "#eef5fb"),
    ]
    for x, y, w, h, text, fill in boxes:
        parts.append(_box(Box(x, y, w, h, text, fill=fill, size=16)))
    parts.append(_line(360, 202, 420, 202))
    parts.append(_line(750, 202, 820, 202))
    parts.append(_line(1140, 202, 1190, 202))
    parts.append(_line(260, 266, 260, 290))
    parts.append(_line(590, 266, 590, 290))
    parts.append(_line(960, 266, 960, 290))
    parts.append(_line(1330, 266, 1330, 290))
    parts.append(_box(Box(210, 400, 1180, 72, "The layers remain modular: UI, orchestration, core pipelines, dataops, models, and deployment are separate.", fill="#ffffff", stroke="#d8dfe8", size=17, weight=500)))
    parts.append(_footer())
    return "".join(parts)


def svg_code_architecture_data_flow() -> str:
    width, height = 1600, 420
    parts = [_header("Code Architecture and Data Flow Map", "End-to-end research-to-deployment flow", width, height)]
    parts.append(_panel(50, 90, 1500, 280, "End-to-end data flow"))
    flow = [
        (120, "Raw images\n+ masks", "#f6ecd7"),
        (350, "Dataset\npreparation", "#eef5fb"),
        (580, "Curated dataset\nwith manifests", "#eef5fb"),
        (830, "Training", "#eef5fb"),
        (1030, "Model checkpoints\n+ logs", "#eef5fb"),
        (1240, "Evaluation\n+ analysis", "#eef5fb"),
        (1450, "Reports\nJSON / HTML / CSV", "#eaf6ea"),
    ]
    for x, text, fill in flow:
        parts.append(_box(Box(x, 170, 160, 72, text, fill=fill, size=16)))
    for x1, x2 in zip([280, 510, 760, 960, 1160, 1370], [350, 580, 830, 1030, 1240, 1450]):
        parts.append(_line(x1, 206, x2, 206))
    parts.append(_footer())
    return "".join(parts)


def svg_code_architecture_runtime_interaction() -> str:
    width, height = 1600, 520
    parts = [_header("Code Architecture and Data Flow Map", "Qt runtime interaction", width, height)]
    parts.append(_panel(50, 90, 1500, 380, "Qt runtime interaction"))
    runtime = [
        (110, "User", "#f6ecd7"),
        (300, "GUI\nmain window", "#eef5fb"),
        (550, "Desktop\nworkflow", "#eef5fb"),
        (800, "Segmentation\npipeline", "#eef5fb"),
        (1050, "Predictor", "#eef5fb"),
        (1250, "Analyzer", "#eef5fb"),
        (1435, "Export\nartifact", "#eaf6ea"),
    ]
    for x, text, fill in runtime:
        parts.append(_box(Box(x, 180, 150, 72, text, fill=fill, size=16)))
    for x1, x2 in zip([260, 480, 730, 980, 1180, 1380], [300, 550, 800, 1050, 1250, 1435]):
        parts.append(_line(x1, 216, x2, 216))
    parts.append(_box(Box(320, 310, 960, 70, "The GUI and the core stay decoupled; the UI only receives status, result metadata, and export locations.", fill="#ffffff", stroke="#d8dfe8", size=16, weight=500)))
    parts.append(_line(1435, 252, 1100, 310))
    parts.append(_line(1050, 252, 800, 310))
    parts.append(_footer())
    return "".join(parts)


def main() -> None:
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    figures = {
        "learning_path.svg": svg_learning_path(),
        "worked_example_conventional_vs_ml.svg": svg_worked_example(),
        "model_selection_decision_tree.svg": svg_model_selection(),
        "conventional_segmentation_pipeline.svg": svg_conventional_pipeline(),
        "gui_model_integration_guide.svg": svg_gui_model_integration(),
        "deployment_productization_master_roadmap.svg": svg_workflow_roadmap(),
        "model_architecture_manuscript_foundation.svg": svg_model_architecture(),
        "code_architecture_map.svg": svg_code_architecture_map(),
        "code_architecture_system_view.svg": svg_code_architecture_system_view(),
        "code_architecture_data_flow.svg": svg_code_architecture_data_flow(),
        "code_architecture_runtime_interaction.svg": svg_code_architecture_runtime_interaction(),
    }
    for name, svg in figures.items():
        (DIAGRAMS / name).write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
