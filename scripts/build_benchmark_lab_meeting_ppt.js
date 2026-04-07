#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

function warnIfSlideHasOverlaps(_slide, _pptx) {
  return;
}

function warnIfSlideElementsOutOfBounds(_slide, _pptx) {
  return;
}

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith("--")) {
      continue;
    }
    out[key.slice(2)] = argv[i + 1];
    i += 1;
  }
  return out;
}

function addFrame(slide, pptx, title, bottomLine) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.6,
    fill: { color: "000000" },
    line: { color: "000000" },
  });
  slide.addText(title, {
    x: 0.2,
    y: 0.12,
    w: 12.8,
    h: 0.3,
    color: "FFFFFF",
    fontFace: "Arial",
    fontSize: 24,
    bold: true,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 6.85,
    w: 13.333,
    h: 0.65,
    fill: { color: "000000" },
    line: { color: "000000" },
  });
  slide.addText(bottomLine, {
    x: 0.2,
    y: 6.98,
    w: 12.8,
    h: 0.28,
    color: "FFFFFF",
    fontFace: "Arial",
    fontSize: 20,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.15,
    y: 0.8,
    w: 2.35,
    h: 5.8,
    fill: { color: "F5F5F5" },
    line: { color: "D0D0D0", pt: 1 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 2.65,
    y: 0.8,
    w: 10.53,
    h: 5.8,
    fill: { color: "FFFFFF" },
    line: { color: "D0D0D0", pt: 1 },
  });
}

function addBullets(slide, bullets) {
  const runs = bullets.map((bullet) => ({
    text: bullet,
    options: { bullet: { indent: 14 } },
  }));
  slide.addText(runs, {
    x: 0.32,
    y: 1.05,
    w: 2.0,
    h: 5.2,
    fontFace: "Arial",
    fontSize: 18,
    color: "111111",
    breakLine: true,
    margin: 0.06,
    valign: "top",
  });
}

function addTextContent(slide, content) {
  const paragraphs = Array.isArray(content.paragraphs) ? content.paragraphs.join("\n\n") : "";
  slide.addText(content.title || "", {
    x: 2.95,
    y: 1.05,
    w: 9.9,
    h: 0.35,
    fontFace: "Arial",
    fontSize: 20,
    bold: true,
    color: "111111",
    margin: 0,
  });
  slide.addText(paragraphs, {
    x: 2.95,
    y: 1.55,
    w: 9.8,
    h: 4.5,
    fontFace: "Arial",
    fontSize: 20,
    color: "222222",
    margin: 0.05,
    valign: "mid",
  });
}

function addTableContent(slide, content) {
  const rows = [content.headers || []].concat(content.rows || []);
  slide.addTable(rows, {
    x: 2.9,
    y: 1.0,
    w: 9.95,
    h: 5.2,
    border: { type: "solid", color: "C8C8C8", pt: 1 },
    fill: "FFFFFF",
    color: "111111",
    fontFace: "Arial",
    fontSize: 18,
    margin: 0.05,
    rowH: 0.55,
    bold: true,
    autoFit: false,
  });
}

function addBarChart(slide, pptx, content) {
  const categories = content.categories || [];
  const values = content.values || [];
  slide.addChart(
    pptx.ChartType.bar,
    [
      {
        name: content.seriesName || content.series_name || "Value",
        labels: categories,
        values,
      },
    ],
    {
      x: 2.95,
      y: 1.0,
      w: 9.9,
      h: 5.2,
      catAxisLabelFontFace: "Arial",
      catAxisLabelFontSize: 16,
      valAxisLabelFontFace: "Arial",
      valAxisLabelFontSize: 16,
      showLegend: false,
      showTitle: false,
      chartColors: ["1F4E79"],
      valAxisMinVal: 0,
      valGridLine: { color: "D9D9D9", pt: 1 },
    }
  );
}

function main() {
  const args = parseArgs(process.argv);
  if (!args.manifest || !args.output) {
    throw new Error("Expected --manifest and --output");
  }
  const manifest = JSON.parse(fs.readFileSync(args.manifest, "utf8"));
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "OpenAI Codex";
  pptx.company = "HydrideSegmentation";
  pptx.subject = manifest.title || "Benchmark results";
  pptx.title = manifest.title || "Benchmark lab meeting";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Arial",
    bodyFontFace: "Arial",
    lang: "en-US",
  };

  (manifest.slides || []).forEach((slideSpec) => {
    const slide = pptx.addSlide();
    addFrame(slide, pptx, slideSpec.title || "", slideSpec.bottom_line || "");
    addBullets(slide, slideSpec.bullets || []);
    const content = slideSpec.content || {};
    if (content.type === "table") {
      addTableContent(slide, content);
    } else if (content.type === "bar_chart") {
      addBarChart(slide, pptx, content);
    } else {
      addTextContent(slide, content);
    }
    warnIfSlideHasOverlaps(slide, pptx);
    warnIfSlideElementsOutOfBounds(slide, pptx);
  });

  const outputPath = path.resolve(args.output);
  return pptx.writeFile({ fileName: outputPath });
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
