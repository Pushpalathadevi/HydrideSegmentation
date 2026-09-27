// Capture web-UI screenshots for the HCI review deck with headless Edge over CDP.
// Usage (web server running on :8077):  node docs/presentations/hci_review/capture_ui.mjs
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "assets");
const base = process.env.MICROSEG_URL || "http://127.0.0.1:8077";
const edge = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const port = 9333;
const profile = mkdtempSync(join(tmpdir(), "hci-cdp-"));
const browser = spawn(edge, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await sleep(500);
  }
  throw new Error("Edge DevTools endpoint not reachable");
}

const ws = new WebSocket(await target());
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
const send = (method, params = {}) => new Promise((resolve) => { const i = ++id; pending.set(i, resolve); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1500, height: 1100, deviceScaleFactor: 1.5, mobile: false });

async function shot(selector, file, pad = 12) {
  const rect = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(selector)}); e.scrollIntoView({block: "start"}); window.scrollBy(0, -220); const r = e.getBoundingClientRect(); return {x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height}; })()`);
  await sleep(600);
  await sleep(600);
const res = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: Math.max(0, rect.x - pad), y: Math.max(0, rect.y - pad), width: rect.w + 2 * pad, height: rect.h + 2 * pad, scale: 1 } });
  writeFileSync(join(out, file), Buffer.from(res.result.data, "base64"));
  console.log("saved", file, Math.round(rect.w), "x", Math.round(rect.h));
}

// 1. Workspace: segment a library image with the HCI on (default) and capture the result panels.
await send("Page.navigate", { url: base + "/" });
await sleep(2500);
await evaluate(`document.getElementById("library-open").click()`);
await sleep(2500);
const picked = await evaluate(`(() => { const b = Array.from(document.querySelectorAll("#library-grid [role=listitem], #library-grid button")).find(x => ((x.title || "") + (x.textContent || "")).includes(${JSON.stringify(process.env.MICROSEG_IMAGE || "q_4_hydrided.png")})); if (!b) return false; b.click(); return true; })()`);
if (!picked) throw new Error("library image not found");
await sleep(1500);
await evaluate(`document.getElementById("run-btn").click()`);
for (let i = 0; i < 180; i++) {
  await sleep(1000);
  const stage = await evaluate(`document.getElementById("progress-stage").textContent`);
  if (/complete/i.test(stage || "")) break;
}
await sleep(1500);
await shot("#hci-fieldset", "ui_hci_controls.png");
await shot("#hci-panel", "ui_hci_panel.png");
await evaluate(`document.querySelector('[data-view="hci_clusters_png_b64"]').click()`);
await sleep(800);
await shot(".viewer", "ui_hci_clusters_view.png");
await evaluate(`document.querySelector('[data-view="hci_curve_png_b64"]').click()`);
await sleep(800);
await shot(".viewer", "ui_hci_curve_view.png");

// 2. Help page: the HCI section with KaTeX-rendered equations.
await send("Page.navigate", { url: base + "/help#hci" });
await sleep(3500);
const section = await evaluate(`(() => { const s = document.querySelector("#hci"); const h = Array.from(s.querySelectorAll("h3")); h[1].scrollIntoView({block: "start"}); window.scrollBy(0, -220); const r0 = h[1].getBoundingClientRect(); const r1 = h[3].getBoundingClientRect(); return {x: r0.left + window.scrollX - 10, y: r0.top + window.scrollY - 10, w: s.getBoundingClientRect().width + 20, h: r1.top - r0.top + 10}; })()`);
await sleep(600);
const res = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: section.x, y: section.y, width: section.w, height: section.h, scale: 1 } });
writeFileSync(join(out, "ui_help_math.png"), Buffer.from(res.result.data, "base64"));
console.log("saved ui_help_math.png");

ws.close();
browser.kill();
process.exit(0);
