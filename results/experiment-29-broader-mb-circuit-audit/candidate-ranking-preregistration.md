# Experiment 29 — Candidate ranking (preregistered, hash-frozen before any leverage simulation)

Inputs: `kc-visual-response.json` (frozen visual gate, seeds 3001–3006, 60/8), `static-leverage-screen.json`,
`kc-mbon-connectivity.json`, `dan-compartment-audit.json`. No dynamic leverage result exists at freeze time.

Rule (from `preregistration.md`): eligible = KC class passes the visual gate AND >= 1 KC->MBON edge AND a
topology-matched DAN exists. Order: visual gate, matched DAN, pooled static MBON input fraction (desc),
KC->MBON |W| (desc), participating KC cells (desc), key. All units are single KC type -> single MBON type
(complexity, #KC classes, #MBON classes tie). Top 10 eligible are tested dynamically; the E26 rows
KCg-d->MBON01 (PAM01) and KCg-d->MBON11 (PPL101) are always run as labelled references.

Visually passing KC classes: KCab-p, KCapbp-ap1, KCapbp-ap2, KCapbp-m
Eligible units: 77; dynamically tested: 10

| rank | KC -> MBON | DAN (topology) | static input % (pooled / median) | KC->MBON abs W | edges | KC cells | tested |
|---|---|---|---|---|---|---|---|
| 1 | KCapbp-ap2->MBON17 | PPL104 | 42.72 / 42.72 | 0.8545 | 285 | 285 | yes |
| 2 | KCapbp-ap2->MBON16 | PPL104 | 41.26 / 41.26 | 0.8252 | 286 | 286 | yes |
| 3 | KCapbp-ap2->MBON13 | PPL105 | 41.01 / 41.01 | 0.8203 | 290 | 290 | yes |
| 4 | KCapbp-ap2->MBON03 | PAM06 | 39.08 / 39.08 | 0.7815 | 577 | 291 | yes |
| 5 | KCapbp-m->MBON13 | PPL105 | 29.89 / 29.89 | 0.5978 | 205 | 205 | yes |
| 6 | KCapbp-m->MBON17 | PPL104 | 29.23 / 29.23 | 0.5847 | 201 | 201 | yes |
| 7 | KCapbp-m->MBON15-like | PPL103 | 28.27 / 28.81 | 1.1306 | 380 | 204 | yes |
| 8 | KCapbp-ap1->MBON16 | PPL104 | 28.17 / 28.17 | 0.5635 | 181 | 181 | yes |
| 9 | KCapbp-ap2->MBON15-like | PPL103 | 28.04 / 27.12 | 1.1217 | 482 | 281 | yes |
| 10 | KCapbp-ap1->MBON28 | PPL104 | 27.61 / 27.61 | 0.5521 | 182 | 182 | yes |
| 11 | KCapbp-m->MBON03 | PAM06 | 24.85 / 24.85 | 0.4970 | 394 | 204 | no |
| 12 | KCab-p->MBON19 | PPL105 | 24.08 / 29.06 | 0.9632 | 214 | 129 | no |
| 13 | KCapbp-ap1->MBON31 | PPL103 | 22.30 / 22.30 | 0.4459 | 197 | 197 | no |
| 14 | KCapbp-ap2->MBON15 | PPL103 | 21.84 / 19.90 | 0.8735 | 339 | 239 | no |
| 15 | KCapbp-ap2->MBON12 | PPL103 | 17.25 / 17.60 | 0.6901 | 581 | 291 | no |
| 16 | KCapbp-ap1->MBON26 | PAM05 | 17.14 / 17.14 | 0.3428 | 196 | 196 | no |
| 17 | KCapbp-ap1->MBON03 | PAM06 | 14.87 / 14.87 | 0.2973 | 260 | 197 | no |
| 18 | KCapbp-ap1->MBON10 | PAM13 | 14.36 / 13.16 | 1.2922 | 530 | 193 | no |
| 19 | KCapbp-m->MBON15 | PPL103 | 13.36 / 13.37 | 0.5343 | 221 | 162 | no |
| 20 | KCapbp-m->MBON19 | PPL105 | 12.95 / 9.05 | 0.5182 | 199 | 154 | no |
| 21 | KCapbp-ap1->MBON12 | PPL103 | 12.92 / 13.49 | 0.5167 | 392 | 197 | no |
| 22 | KCapbp-ap2->MBON31 | PPL103 | 12.12 / 12.12 | 0.2424 | 291 | 291 | no |
| 23 | KCapbp-ap2->MBON28 | PPL104 | 11.10 / 11.10 | 0.2219 | 207 | 207 | no |
| 24 | KCapbp-ap2->MBON10 | PAM13 | 10.62 / 11.54 | 0.9554 | 495 | 253 | no |
| 25 | KCapbp-ap2->MBON09 | PAM13 | 10.49 / 10.94 | 0.4196 | 583 | 291 | no |
| 26 | KCab-p->MBON06 | PAM10 | 10.25 / 10.25 | 0.2050 | 129 | 129 | no |
| 27 | KCab-p->MBON23 | PPL105 | 9.79 / 9.79 | 0.1958 | 124 | 124 | no |
| 28 | KCapbp-ap1->MBON04 | PAM05 | 8.87 / 8.87 | 0.1774 | 205 | 196 | no |
| 29 | KCapbp-ap2->MBON04 | PAM05 | 8.78 / 8.78 | 0.1755 | 304 | 289 | no |
| 30 | KCab-p->MBON07 | PAM11 | 8.72 / 8.59 | 0.3488 | 258 | 129 | no |

References (always run):

- KCg-d->MBON01 (reference DAN PAM01; topology DAN PAM01): static 7.64 %, overall rank 130, eligible False, in top 10: False
- KCg-d->MBON11 (reference DAN PPL101; topology DAN PPL101): static 6.86 %, overall rank 131, eligible False, in top 10: False
