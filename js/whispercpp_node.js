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

app.registerExtension({
    name: "WhisperCPP.AdvancedSettings",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WhisperCPPNode") {
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
});
