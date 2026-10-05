# Phase 1: feasibility protocol

Goal: decide, with measurements from the two reference machines, whether the
mesh-first plan works without NVIDIA hardware. See the Phase 1 section of
the [roadmap](../ROADMAP.md) for the questions and exit decisions. How to
install and run the tools: [`tools/feasibility/README.md`](../../tools/feasibility/README.md).

## 1. Capture the datasets

Use one object for datasets 1 to 4 so results are comparable: something
around 10 to 25 cm, matte, with visible texture (a painted figurine, a shoe,
a carved wooden object). Put it on a patterned surface (newspaper, a printed
noise pattern, a textured placemat). Diffuse light, no harsh shadows. Phone
camera, fixed zoom (1x), no portrait mode.

| # | Folder | How to shoot |
|---|---|---|
| 1 | `~/ez2d-data/walkaround/images` | 40 to 60 photos walking around the object: one ring at about 30° above the table, one at about 60°. Each photo should overlap its neighbours by about two thirds. |
| 2 | `~/ez2d-data/turntable/images` | Object on a turntable (or a lazy Susan, or rotate it by hand on a sheet of paper), phone on a stand, 36 photos at roughly 10° steps, then raise the phone and do a second ring. |
| 3 | `~/ez2d-data/flipped/images/upright` and `.../flipped` | Dataset 1 again, then turn the object upside down and capture the same way. Phone file numbering keeps names unique, which the scripts need. |
| 4 | a video file | 30 to 60 s slow orbit around the object at two heights. Then: `bench.py frames video.mp4 ~/ez2d-data/video/images --count 120` |
| 5 | `~/ez2d-data/hard/images` | Something that should struggle: a glossy mug, a plain white object, or shallow depth of field (close-up). 40 photos as in dataset 1. |

Then create masks for each set and check the preview sheet; tinted red means
background:

```sh
B="uv run --group feasibility tools/feasibility/bench.py"
$B masks ~/ez2d-data/walkaround/images ~/ez2d-data/walkaround/masks
```

If masks are wrong for some images, write that down: mask quality is one of
the Phase 1 questions. Delete a mask file to make that image unmasked.

## 2. Run the plans

On the **7900 GRE machine**, run all of them:

```sh
for p in tools/feasibility/plans/[1-5]-*.toml; do $B run "$p"; done
$B report
```

This can take hours; OpenMVS level 0 and RefineMesh are the slowest
variants. Runs that finished are skipped when you start again.

On the **M1 (8 GB)**, start with a subset and stop variants that swap
heavily for a long time (Ctrl+C, then `--only` the rest):

```sh
$B run tools/feasibility/plans/1-walkaround.toml --only sift1600 sift3200 lvl2 lvl1 brush-1024
$B run tools/feasibility/plans/2-turntable.toml
$B report
```

Use the same `--machine` name for every run on one machine (it defaults to
the hostname).

## 3. Record what you see

Open the outputs (the paths are in each `run.json` under `metrics.result`):
the textured mesh in MeshLab or Blender, the splat `.ply` in Brush's viewer
(`brush_app --with-viewer`, then open the file) or another splat viewer.

Write your observations into [`FINDINGS.md`](FINDINGS.md): the numbers say
how long and how much memory; only you can say whether the mesh is good
enough. Screenshots go in `docs/feasibility/img/` (keep them small, under
about 300 KB each).

## 4. Commit

Commit `docs/feasibility/results/<machine>.md`, `<machine>.json`,
`FINDINGS.md` and the screenshots. Don't commit the run folders or the
photos; the photo sets are worth keeping somewhere safe, since they become
the regression datasets later.
