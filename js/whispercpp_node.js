import { app } from "/scripts/app.js";

// Widgets hidden under show_advance_cpp
const CPP_WIDGETS = [
    "sampling_strategy","best_of","beam_size","patience",
    "temperature","temperature_inc","max_initial_ts","length_penalty",
    "n_max_text_ctx","offset_ms","duration_ms",
    "no_context","single_segment","no_timestamps","max_tokens","max_len","split_on_word",
    "token_timestamps","thold_pt","thold_ptsum",
    "suppress_blank","suppress_nst","suppress_regex",
    "entropy_thold","logprob_thold","no_speech_thold",
    "initial_prompt","carry_initial_prompt",
    "audio_ctx","debug_mode","print_special","print_progress",
    "tdrz_enable","grammar_penalty",
    "flash_attn","gpu_device",
    "dtw_aheads_preset","dtw_n_top",
];

// Widgets hidden under show_advance_ext
const EXT_WIDGETS = [
    "separate_model","separate_chunk_size","separate_overlap",
    "align_model","return_char_alignments",
    "diarize",
];

// ── Issue #16: language dropdown pollution ─────────────────────────────────
// Other extensions (e.g. MiniMax Music Production Toolkit) inject their own
// labels - `日本語 (Japanese)`, `Multilingual / mixed` - into combo widgets that
// share the widget name `language`. ComfyUI then rejects the prompt with
// "Value not in list" because the backend never offered those values.
// Canonical values: "None"/"auto", 2-3 letter codes, Title Case names.
// tests/test_languages.py compiles this exact pattern against every value the
// backend offers - if you change it, change it there too.
const LANG_VALUE_RE = /^(None|auto|[a-z]{2,3}|[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)$/;
const LANG_PAREN_RE = /[（(]([^()]+)[)）]/;

// Map a foreign/legacy selector onto a value the backend accepts.
function toCanonicalLang(value, options) {
    if (typeof value !== "string") return "None";
    if (LANG_VALUE_RE.test(value)) return value;

    const candidates = [];
    const paren = value.match(LANG_PAREN_RE);
    if (paren) {
        candidates.push(paren[1]);
        candidates.push(value.replace(LANG_PAREN_RE, ""));
    }
    candidates.push(value);
    for (const cand of candidates) {
        const t = (cand || "").trim();
        if (!t) continue;
        const low = t.toLowerCase();
        // Only ever adopt a value the backend accepts, even if this list has
        // not been filtered yet.
        const hit = options.find(o => typeof o === "string" && LANG_VALUE_RE.test(o) && o.toLowerCase() === low);
        if (hit) return hit;
    }
    // Unknown label ("Multilingual / mixed", garbage from another extension,
    // ...): fall back to auto-detect instead of submitting an invalid value.
    return "None";
}

// Drop foreign options and fix the current value of the language widget.
function sanitizeLangWidget(node) {
    const w = node?.widgets?.find(x => x.name === "language");
    if (!w) return;
    const opts = w.options?.values;
    if (Array.isArray(opts) && opts.some(o => typeof o !== "string" || !LANG_VALUE_RE.test(o))) {
        // Mutate in place: values is shared with the node definition, so this
        // also heals the list for nodes created later.
        for (let i = opts.length - 1; i >= 0; i--) {
            if (typeof opts[i] !== "string" || !LANG_VALUE_RE.test(opts[i])) opts.splice(i, 1);
        }
    }
    const list = Array.isArray(w.options?.values) ? w.options.values : [];
    const mapped = toCanonicalLang(w.value, list);
    if (mapped !== w.value) w.value = mapped;
}

app.registerExtension({
    name: "WhisperCPP.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperCPPNode") {
            // Widget values are restored by onConfigure, so sanitize there too.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                sanitizeLangWidget(this);
            };

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onCreated?.apply(this, arguments);

                const cppToggle = this.widgets.find(w => w.name === "show_advance_cpp");
                const extToggle = this.widgets.find(w => w.name === "show_advance_ext");

                const cppRefs = CPP_WIDGETS.map(n => this.widgets.find(w => w.name === n)).filter(Boolean);
                const extRefs = EXT_WIDGETS.map(n => this.widgets.find(w => w.name === n)).filter(Boolean);

                const setup = (toggle, refs) => {
                    if (!toggle) return;

                    // Captures the user's manually-set node width so toggles never reset it.
                    // Initialized from .size[0] on first toggle; updated on every resize.
                    // Number.isFinite guards: a non-finite width (NaN/undefined) would
                    // serialize to null in the workflow JSON (issue #15: "size": [null, N])
                    // and leave the node unselectable after reload.
                    let savedWidth = (this.size && Number.isFinite(this.size[0]) && this.size[0] > 0)
                        ? this.size[0]
                        : 400;

                    const update = (show) => {
                        if (!Number.isFinite(savedWidth) || savedWidth <= 0) {
                            savedWidth = (this.size && Number.isFinite(this.size[0]) && this.size[0] > 0)
                                ? this.size[0]
                                : 400;
                        }

                        if (!show) {
                            this.widgets = this.widgets.filter(w => !refs.includes(w));
                        } else {
                            const toAdd = refs.filter(w => !this.widgets.includes(w));
                            if (toAdd.length) {
                                const idx = this.widgets.indexOf(toggle);
                                this.widgets.splice(idx + 1, 0, ...toAdd);
                            }
                        }

                        // Only height changes – width stays at what the user chose.
                        // Validate both: computeSize() can return non-finite values during
                        // workflow restoration, and assigning them corrupts size on disk.
                        const newSize = this.computeSize();
                        const safeWidth = (Number.isFinite(savedWidth) && savedWidth > 0)
                            ? savedWidth
                            : 400;
                        const safeHeight = (newSize && Number.isFinite(newSize[1]))
                            ? newSize[1]
                            : 200;
                        this.size = [safeWidth, safeHeight];
                        this.setDirtyCanvas(true, true);
                    };

                    toggle.callback = (v) => update(v);

                    // Track manual resizes so the user's width is always honoured.
                    // Only accept sane values – a non-finite/zero width here is how
                    // [null, N] got written to workflow JSON in the first place.
                    const origResize = this.onResize;
                    this.onResize = function (w, h) {
                        if (Number.isFinite(w) && w > 0) savedWidth = w;
                        if (origResize) return origResize.apply(this, arguments);
                    };

                    setTimeout(() => update(toggle.value), 10);
                };

                setup(cppToggle, cppRefs);
                setup(extToggle, extRefs);

                // ── Language widget (issue #16) ──
                const langNode = this;
                sanitizeLangWidget(this);
                const langW = this.widgets.find(w => w.name === "language");
                if (langW && !langW.__whisperLangHooked) {
                    langW.__whisperLangHooked = true;
                    const origCb = langW.callback;
                    // Re-sanitize at interaction time: another extension may
                    // have polluted the options after this node was created.
                    langW.callback = function (value, ...rest) {
                        sanitizeLangWidget(langNode);
                        const list = Array.isArray(langW.options?.values) ? langW.options.values : [];
                        const canon = toCanonicalLang(value, list);
                        langW.value = canon;
                        const ret = origCb ? origCb.call(this, canon, ...rest) : undefined;
                        langW.value = canon; // LiteGraph may assign after the callback
                        return ret;
                    };
                }

                // ── Mutual exclusion: DTW ↔ Alignment ──
                const dtwW = this.widgets.find(w => w.name === "dtw_token_timestamps");
                const alignW = this.widgets.find(w => w.name === "align");

                if (dtwW && alignW) {
                    dtwW.callback = (v) => {
                        if (v) {
                            alignW.value = false;
                            if (alignW.callback) alignW.callback(false);
                        }
                    };
                    // Wrap align callback too
                    const origAlignCb = alignW.callback;
                    alignW.callback = (v) => {
                        if (v) {
                            dtwW.value = false;
                            if (dtwW.callback) dtwW.callback(false);
                        }
                        if (origAlignCb) origAlignCb(v);
                    };
                }
            };
        }
    },

    // Widget values from a saved workflow are fully restored once the graph is
    // loaded - run one last pass so a workflow saved while the dropdown was
    // polluted cannot re-submit a foreign value.
    async loadedGraph(graph) {
        const nodes = graph?._nodes ?? [];
        for (const n of nodes) {
            if (n?.type === "WhisperCPPNode" || n?.comfyClass === "WhisperCPPNode") {
                sanitizeLangWidget(n);
            }
        }
    },
});
