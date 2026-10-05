// Runtime check for the language sanitizers in js/whispercpp_node.js (issue #16).
// Run with: node tests/check_lang_sanitize.mjs
// The node file imports ComfyUI's browser-only app module and registers the
// extension at load time, so both are stripped before evaluating the helpers.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const jsPath = path.resolve(here, "..", "js", "whispercpp_node.js");
let src = fs.readFileSync(jsPath, "utf8");

src = src.replace(/^\s*import\s+[^\n]*$/m, "");
const regIdx = src.indexOf("app.registerExtension");
if (regIdx < 0) {
    console.error("FAIL: app.registerExtension not found in whispercpp_node.js");
    process.exit(1);
}
src = src.slice(0, regIdx);
src += "\nexport { LANG_VALUE_RE, toCanonicalLang, sanitizeLangWidget };\n";

const mod = await import(
    "data:text/javascript;base64," + Buffer.from(src, "utf8").toString("base64")
);
const { LANG_VALUE_RE, toCanonicalLang, sanitizeLangWidget } = mod;

// Backend values the dropdown may legitimately hold (see whispercpp.languages).
const CLEAN_OPTIONS = ["None", "auto", "en", "ja", "fr", "ht", "Japanese", "French", "Haitian Creole"];
const POLLUTED = ["日本語 (Japanese)", "Multilingual / mixed"];
// Values injected by MiniMax Music Production Toolkit, per issue #16.
const OPTIONS = [...CLEAN_OPTIONS, ...POLLUTED];

let failures = 0;
function check(cond, msg) {
    if (cond) return;
    failures++;
    console.error(`FAIL: ${msg}`);
}

// ── toCanonicalLang ──
check(toCanonicalLang("日本語 (Japanese)", OPTIONS) === "Japanese",
    `日本語 (Japanese) -> ${toCanonicalLang("日本語 (Japanese)", OPTIONS)}`);
check(toCanonicalLang("Multilingual / mixed", OPTIONS) === "None",
    `Multilingual / mixed -> ${toCanonicalLang("Multilingual / mixed", OPTIONS)}`);
check(toCanonicalLang("ja", OPTIONS) === "ja", "canonical code must pass through");
check(toCanonicalLang("None", OPTIONS) === "None", "'None' must pass through");
check(toCanonicalLang("auto", OPTIONS) === "auto", "'auto' must pass through");
check(toCanonicalLang("Japanese", OPTIONS) === "Japanese", "name must pass through");
check(toCanonicalLang(" Klingon ", OPTIONS) === "None", "unknown selector falls back to auto");
check(toCanonicalLang(12345, OPTIONS) === "None", "non-string falls back to auto");

// ── sanitizeLangWidget: cleans options AND the current value ──
const node = {
    widgets: [
        {
            name: "language",
            value: "日本語 (Japanese)",
            options: { values: [...OPTIONS] },
        },
    ],
};
sanitizeLangWidget(node);
const w = node.widgets[0];
check(w.value === "Japanese", `widget value -> ${w.value}`);
check(w.options.values.every(v => typeof v === "string" && LANG_VALUE_RE.test(v)),
    "polluted options must be filtered out");
for (const label of POLLUTED) {
    check(!w.options.values.includes(label), `${label} still in options`);
}
check(CLEAN_OPTIONS.every(v => w.options.values.includes(v)),
    "legit options must survive the filter");

// ── interactive path: value picked from a still-polluted dropdown ──
const node2 = { widgets: [{ name: "language", value: "en", options: { values: [...OPTIONS] } }] };
sanitizeLangWidget(node2); // only cleans; simulate a pick afterwards
const w2 = node2.widgets[0];
w2.value = "Multilingual / mixed";
const mapped = toCanonicalLang(w2.value, w2.options.values);
check(mapped === "None", `picked Multilingual / mixed -> ${mapped}`);

// ── value restored from a polluted workflow JSON ──
const node3 = { widgets: [{ name: "language", value: "Multilingual / mixed", options: { values: [...CLEAN_OPTIONS] } }] };
sanitizeLangWidget(node3);
check(node3.widgets[0].value === "None", `restored workflow value -> ${node3.widgets[0].value}`);

if (failures) {
    console.error(`${failures} check(s) failed`);
    process.exit(1);
}
console.log("OK");
