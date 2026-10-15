# INIT02: loading improved, instruction-dependent preservation failed

Adding the new loading examples produced 5/8 strict loading successes, compared with 0/8 for the unchanged parent and matched control. The same model disturbed all 8 completed-feeding scenes, while parent/control preserved 8/8. The earlier five passing skill cases remained passing, but both continued models lost pickup on two cases whose final result already failed.

The added movement was useful, but it was not selected reliably by the requested task. This model does not pass the planned advancement gate. No optional development trials or extra training were run. See [full report](INIT02_report.md), [methods](METHODS.md), and comparisons.csv for every fixed case.

Saved-packet diagnostics isolate instruction changes at identical initial image/mask/proprio/domain inputs and noise. All 24 first chunks change with text; T mean XYZ difference is 1.385 mm over the first 30 predicted commands. This shows text sensitivity, not correct task understanding. Under feeding requests T touched 8/8, lifted 7/8, and reached the opposite loading endpoint in 5/8; those remain feeding failures.

This is local expert-generated robot supervision at personal-data-derived scenarios. It does not recover human motor commands, establish broad generalisation, or test numeral recognition. The programme remains active; the next experiment is being selected with the planner.

Visual evidence: [loading outcomes](media-preview/initiation.jpg), [preservation failures](media-preview/preservation.jpg), [earlier feeding](media-preview/prior_feeding.jpg), [earlier loading](media-preview/prior_loading.jpg), and [matched motion](media-preview/matched-loading-filmstrip.jpg). Full73-second simulation videos are in the verified Mint archive; their nominal encoded durations include boundary capture frames.
