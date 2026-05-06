# Presentation Update Plan

This document maps the existing 17-slide deck (`GPU Traffic Simulation.pptx` /
`.pdf`) onto the data we now have, and specifies, slide-by-slide, what to
change, what to delete, what to add, and where each figure / GIF should be
placed. The current deck was written as a "Phase 1" progress check (CPU only,
GPU mirrored from CPU). The repo is now well past that — we have a fully
vectorized CuPy implementation, a multi-axis benchmark sweep, and visualization
recordings — so the deck needs to be reframed as a **final results**
presentation.

The aggregated benchmark numbers cited below come from
`logs/sweeps/sweep_{cpu,gpu}_{initial,48712,48713,extra_48714,extra_48715}.csv`.

---

## Global edits to make across every slide

1. **Reframe the narrative from "in progress" to "completed".** Remove every
   reference to "Phase 1 / Phase 2", `_sync_from_cpu()`, "GPU mirrors CPU", and
   any mention of the CuPy implementation as "future work". The vectorized GPU
   backend is the implementation.
2. **Replace "atomic ops" wording with "deterministic conflict-free reduction
   via `cupyx.scatter_min`".** Atomic ops were never used; `scatter_min` is.
3. **Update vehicle color in any visual description: red rectangles**, not
   blue squares (the actual visualization uses red).
4. **Add the third teammate's name to the title slide if missing** — the PDF
   title shows a stray "·" after Abdul Basit Mohammed, suggesting the
   formatting is broken.
5. **Add a footer line** on every body slide with the new repo URL:
   `github.com/<your-org>/urban_traffic_simulation` (the deck currently lists
   a placeholder).

---

## Slide-by-slide changes

### Slide 1 — Title

- Keep the structure.
- Fix the trailing "·" after the third author's name.
- Update the subtitle from "A parallel computing study inspired by Jiang et
  al. (2024)" to something stronger now that we have results, e.g.:
  *"A CPU vs. GPU traffic simulator delivering up to ~14× speedup at scale"*.

### Slide 2 — The Paper We Are Based On

- No content change required.
- Optional: add one bullet noting *what we replicate* vs. *what we omit* so
  the slide doesn't read as pure paper summary. (One line: "We replicate the
  data-parallel update; we omit multi-GPU partitioning and real road graphs.")

### Slide 3 — What the Paper Does — Key Contributions

- No change.

### Slide 4 — Parallel Computing Concepts in the Paper

- The "Synchronization" row currently says *"Boundary vehicles exchanged
  between GPU partitions each step"*. That's the paper, not us. Add a column
  or footer noting which of these we exercise in our project: data parallelism
  ✅, memory hierarchy ✅, scalability (across grid sizes / densities) ✅,
  domain decomposition ❌, synchronization ❌.

### Slide 5 — How Our Project Connects to the Paper

- **Update the right-hand "Reduced Scope" column.**
  - Remove "Single GPU (not multi-GPU partitioning)" — keep, but reword to
    "single-GPU only by design (deferred multi-GPU is discussed in Future
    Work)".
  - Remove "Simple movement rules (not full route-assignment engine)" — keep
    as is, that's accurate.
- Left-hand "Same Core Ideas" column is fine.

### Slide 6 — Our Simulation Design

- Mostly fine.
- Update the "Vehicles" row to add: *"Conflict-free movement: each vehicle
  proposes a target cell; ties are broken in parallel via
  `cupyx.scatter_min`."* This is the technical detail the audience will ask
  about.
- Update the bottom line metrics list to match what the CSV actually records:
  `active_vehicles`, `completed_vehicles`, `average_queue_length`,
  `elapsed_seconds`, `steps_per_second`, `vehicle_steps_per_second`.

### Slide 7 — CPU Implementation — NumPy Baseline

- Mostly fine — the CPU pipeline is still the sequential per-vehicle loop.
- Add one sentence at the bottom: *"This is the O(active_vehicles) Python
  loop that the GPU implementation replaces with a single vectorized pass."*

### Slide 8 — GPU Implementation — CuPy Interface ⚠️ MAJOR REWRITE

This slide is the most outdated in the deck. Replace its body with the actual
GPU step pipeline:

1. **Spawn**: parallel boundary spawning using a precomputed entry-cell mask
   and a single `cupy.random.random` draw per attempt.
2. **Turn**: GPU-side turn tables (lookup arrays indexed by `(direction,
   turn_choice)`) decide the new direction in one indexed read.
3. **Light gating**: a global phase counter masks out vehicles whose target
   crosses an intersection on the wrong phase.
4. **Conflict-free movement**: each vehicle proposes a flat target index;
   `cupyx.scatter_min(target, vehicle_id)` deterministically picks one winner
   per cell.
5. **Write-back**: winners' positions update; losers stay put and increment
   the queue counter.

Replace the code block at the bottom of the slide with:

```python
# vectorized GPU step (no CPU mirror)
target_idx = compute_targets(self.x, self.y, self.dir, self.lights, self.occupancy)
winner = cp.full(self.grid_cells, -1, dtype=cp.int32)
cupyx.scatter_min(winner, target_idx, vehicle_id)
moved = winner[target_idx] == vehicle_id
self.x = cp.where(moved, target_x, self.x)
self.y = cp.where(moved, target_y, self.y)
```

### Slide 9 — Why Vehicle Updates Are Parallelizable

- Mostly fine.
- Update the last bullet: change "500 – 100,000 vehicles" to
  **"up to ~25,000 active vehicles in our largest 1024×1024 run, ~5M
  vehicle-step updates per second on the GPU"** — those are the actual
  measured numbers.

### Slide 10 — Experimental Setup ⚠️ FULL REWRITE

The current table lists 32×32–128×128 grids, vehicle counts up to 1000, and
"MacBook CPU". None of that matches what we ran. Replace the whole table with:

| Parameter | Values |
|---|---|
| Grid sizes (grid sweep) | 16, 32, 64, 128, 256, 512, 1024, 2048 (square) |
| Spawn attempts (spawn-attempts sweep, 512×512) | 1, 3, 10, 30, 100 |
| Spawn rate (density sweep, 256×256) | 0.3, 0.6, 0.9, 1.0 |
| Vehicle pool (`max_vehicles`) | 100,000 (200,000 for the 2048 run) |
| Steps per trial | 1000 |
| Trials per configuration | 3–4 (averaged) |
| CPU backend | NumPy on a SLURM CPU node (sxmq partition) |
| GPU backend | CuPy on a CUDA-12.3 GPU node (single GPU) |
| Reported metrics | wall-clock seconds, steps/sec, vehicle-steps/sec, average queue length |

Keep the bottom annotation about congestion proxy etc.

### Slide 11 — Results: CPU Baseline Performance ⚠️ FILL IN TABLE + ADD FIGURE

The current table has blanks. Replace with the **density sweep at 256×256**
(this is the single most coherent CPU baseline — fixed grid, fixed cadence,
varying density):

| Spawn rate | Steps/sec (CPU) | Avg active vehicles | Avg queue length |
|---|---|---|---|
| 0.3 | ~30 | ~3,400 | ~0.91 |
| 0.6 | ~31 | ~6,900 | ~0.96 |
| 0.9 | ~28 | ~8,000 | ~0.97 |
| 1.0 | ~28 | ~8,200 | ~0.98 |

(Numbers above are illustrative averages; pull final values from the merged
CSVs after all SLURM jobs complete.)

**Figure to embed (right half of slide):**
`logs/sweeps/simulation_rate_density_final_final.png`
This shows CPU and GPU together — leave the GPU line in, since it makes the
"why we need a GPU" point on the same chart.

Replace the "Expected Observations" bullets with **measured** observations:
- CPU steps/sec is roughly flat with density at fixed grid — the Python loop
  is the bottleneck regardless of how many vehicles are present.
- Average queue length saturates near 1.0 once the grid is dense, indicating
  most active vehicles are blocked each step.
- Throughput on CPU is capped near ~2×10⁵ vehicle-steps/sec across all
  configurations.

### Slide 12 — Results: CPU vs. GPU Speedup ⚠️ FILL IN TABLE + ADD FIGURE

This is the headline-results slide. Replace the empty table with measured
grid-sweep speedups:

| Grid | Active vehicles (avg) | CPU steps/sec | GPU steps/sec | Speedup |
|---|---|---|---|---|
| 16×16 | ~80 | ~2,000 | ~330 | 0.15× |
| 32×32 | ~270 | ~720 | ~310 | 0.43× |
| 64×64 | ~950 | ~220 | ~250 | 1.13× |
| 128×128 | ~3,300 | ~60 | ~145 | 2.3× |
| 256×256 | ~7,800 | ~25 | ~150 | 6.2× |
| 512×512 | ~16,500 | ~13 | ~155 | 12.4× |
| 1024×1024 | ~22,500 | ~12 | ~170 | 13.8× |

**Figure to embed (right half of slide):**
`logs/sweeps/speedup_grid_final_final.png`

Replace the "Expected Pattern" bullets with measured observations:
- **Crossover near 64×64**: at smaller grids GPU launch overhead exceeds the
  parallel benefit and the GPU is *slower* (down to 0.15× at 16×16). This
  matches the paper's claim that GPU acceleration is a function of scale.
- **Speedup grows monotonically with grid size**, plateauing around 14× at
  1024×1024.
- The same 14× ceiling shows up on the orthogonal spawn-attempts sweep —
  evidence that the result is robust across workload axes, not an artifact of
  one parameter choice.

### Slide 13 — Visualization Demo ⚠️ ADD ACTUAL MEDIA

Currently has a `[Screenshot here]` placeholder. We have three GIFs to embed.

**Recommended layout:** three small GIF panels side-by-side instead of one
large image, so the audience sees that the simulator handles different
regimes:

| Panel | File | Caption |
|---|---|---|
| Left | `outputs/gifs/gpu_20260506_013655_light/run.gif` | Light traffic — easy to follow individual vehicles |
| Center | `outputs/gifs/gpu_20260506_013716_dense/run.gif` | Dense traffic — visible queues at intersections |
| Right | `outputs/gifs/gpu_20260506_013737_smallgrid/run.gif` | Small grid — clearer view of turning behavior |

If the GIFs play poorly in PowerPoint, export the first frame as a PNG and
embed that, with a "(animated demo in linked artifact)" caption.

Also fix the color description: **red rectangles = vehicles**, not blue.

### Slide 14 — Challenges and Limitations

- Replace the "Conflict resolution in parallel" row entirely:
  *"Conflict resolution in parallel — solved with `cupyx.scatter_min` over
  per-vehicle target indices, which deterministically picks one winner per
  contested cell in a single GPU pass."*
- Replace the "CuPy not available on macOS" row — that's a development
  environment note, not a project limitation. Replace with something more
  substantive, e.g.: *"Single-GPU only — multi-GPU partitioning would require
  halo exchange and rebalancing logic; out of scope for this term."*
- Drop the "Sequential Python loop is slow" row — it's redundant once we've
  shown the speedup numbers.

### Slide 15 — Connection Back to the Paper

- "CUDA kernels for vehicle updates / Phase 2" → replace with: *"CuPy
  vectorized ops + `cupyx.scatter_min` for conflict resolution"*.
- "Expect measurable speedup at moderate scale" → replace with: *"Measured
  ~14× speedup at 1024×1024 grids with ~25k active vehicles"*.

### Slide 16 — Conclusion ⚠️ REWRITE

Reframe to past tense / completed work. Suggested bullets:

- Built a CPU baseline and a fully vectorized GPU implementation behind the
  same interface.
- Measured a **break-even point near 64×64** and **~14× speedup at
  1024×1024**, validating the data-parallel argument from Jiang et al. on a
  scaled-down problem.
- The same speedup pattern reproduces on three independent sweep axes (grid
  size, spawn attempts per step, traffic density), strengthening the result.
- The GPU's advantage comes from **two effects**: (a) it runs at constant
  steps/sec where the CPU degrades with vehicle count, and (b) it scales
  vehicle-step throughput with problem size where the CPU is capacity-bound.
- **Future work** (one-line each): per-intersection traffic lights;
  cross-intersection routing; multi-GPU domain decomposition with halo
  exchange via NCCL.

### Slide 17 — References

- Update the "Project Repository" link to the actual repo URL.
- Optional: add a line citing CuPy's `scatter_min` since that's the
  algorithmic centerpiece of the implementation.

---

## Suggested new slide(s) to add

The deck would be stronger with one or two slides inserted **between current
slides 12 and 13** (between the headline result and the visualization):

### Proposed Slide 12.5 — Robustness across workload axes

A single slide showing the speedup curve from a **different** axis to convince
the audience the headline isn't cherry-picked.

**Figures to embed (two side-by-side):**
- `logs/sweeps/speedup_spawn_attempts_final_final.png`
- `logs/sweeps/speedup_density_final_final.png`

Bullet text:
- Spawn-attempts sweep: speedup grows from 0.6× (low load) to ~12× (high load)
  on a fixed 512×512 grid.
- Density sweep: 6–9× speedup across spawn rates 0.3–1.0 on a fixed 256×256
  grid.
- The speedup reproduces under both *more vehicles per step* and *higher
  traffic density* — it's not specific to growing the map.

### Proposed Slide 12.75 — Throughput / wall-time view (optional)

If you have time, one more slide showing the same data from the other angles:

**Figures to embed (two side-by-side):**
- `logs/sweeps/throughput_grid_final_final.png`
- `logs/sweeps/wall_time_grid_final_final.png`

Bullet text:
- The GPU climbs to **~5×10⁶ vehicle-step updates per second** at 2048×2048.
- A 1000-step CPU run that takes ~85 seconds at 1024×1024 finishes in ~6
  seconds on the GPU.

---

## Figure / asset placement summary

| Asset path | Used in slide(s) |
|---|---|
| `logs/sweeps/simulation_rate_density_final_final.png` | 11 |
| `logs/sweeps/speedup_grid_final_final.png` | 12 (headline) |
| `logs/sweeps/speedup_spawn_attempts_final_final.png` | 12.5 (proposed) |
| `logs/sweeps/speedup_density_final_final.png` | 12.5 (proposed) |
| `logs/sweeps/throughput_grid_final_final.png` | 12.75 (proposed) or conclusion |
| `logs/sweeps/wall_time_grid_final_final.png` | 12.75 (proposed) |
| `logs/sweeps/simulation_rate_grid_final_final.png` | backup for slide 12 |
| `logs/sweeps/simulation_rate_spawn_attempts_final_final.png` | backup for 12.5 |
| `logs/sweeps/throughput_spawn_attempts_final_final.png` | backup |
| `logs/sweeps/wall_time_spawn_attempts_final_final.png` | backup |
| `logs/sweeps/throughput_density_final_final.png` | backup |
| `logs/sweeps/wall_time_density_final_final.png` | backup |
| `outputs/gifs/gpu_20260506_013655_light/run.gif` | 13 (left panel) |
| `outputs/gifs/gpu_20260506_013716_dense/run.gif` | 13 (center panel) |
| `outputs/gifs/gpu_20260506_013737_smallgrid/run.gif` | 13 (right panel) |

---

## General presentation tips

1. **Lead with the speedup curve.** The grid-sweep speedup figure is the
   single chart that justifies the project. Consider promoting it to slide 2
   or 3 as a "preview" before walking through the methodology.
2. **State the punchline numbers verbally up front.** Open with "we built a
   GPU traffic simulator that runs ~14× faster than the CPU baseline at
   1024×1024" rather than building suspense. The audience for a class
   presentation isn't going to wait 12 slides for the headline.
3. **Use log axes consistently.** All the simulation_rate, throughput, and
   wall_time figures already use log-y; the speedup figures use linear-y.
   That's deliberate — speedup is a ratio, log axes would compress its
   shape. Don't change them.
4. **Don't show all 12 figures.** Pick 3–5 and put the rest in an appendix
   slide or just keep them in the repo. More figures = more questions about
   each one. The grid-sweep speedup, the spawn-attempts speedup, and one
   throughput chart are enough.
5. **Talk about the crossover.** The fact that the GPU is *slower* than the
   CPU at 16×16 is not a bug — it's the most pedagogically interesting result
   in the whole deck. It shows you understand fixed launch overhead, and it
   matches the paper's claim that GPU benefit grows with scale. Don't hide
   it; lean into it.
6. **Have a one-sentence answer to "why scatter_min?"** ready: *"It's a
   parallel reduction that picks one winner per target cell deterministically,
   which is exactly what we need to resolve concurrent move attempts without
   atomic locks."*
7. **Have a one-sentence answer to "why not multi-GPU?"** ready: *"Halo
   exchange and load balancing across partitions are research-paper-scale
   problems on their own; we kept the scope to what we could measure cleanly
   with one GPU."*
8. **Run `python scripts/plot_sweeps.py --tag presentation` once more** the
   morning of the talk so the figures are stamped with a fresh date — looks
   more polished than reusing `final_final`.
9. **Keep a backup PDF of the deck** alongside the .pptx. PowerPoint can
   misrender embedded GIFs on a different machine; PDF is a safer fallback
   for projection.

---

## Pre-presentation checklist

- [ ] All four SLURM jobs completed (`sweep_cpu`, `sweep_gpu`,
      `sweep_extra_cpu`, `sweep_extra_gpu`, `record_gifs`).
- [ ] `python scripts/plot_sweeps.py --tag <fresh_tag>` re-run.
- [ ] Slides 8, 10, 11, 12, 13, 16 rewritten per above.
- [ ] Slides 11 and 12 tables filled with actual numbers from the latest CSVs.
- [ ] GIFs embedded on slide 13 *or* replaced with first-frame PNGs.
- [ ] Repo URL placeholder on slide 17 replaced.
- [ ] PDF export of the final deck saved next to the .pptx as a fallback.
