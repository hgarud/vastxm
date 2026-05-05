# vastXM

Tiny CLI wrapper over the [vast.ai](https://vast.ai) CLI for launching one-off GPU training jobs.

`vastxm launch` provisions an on-demand GPU instance, ships your project code to it, runs a command you specify, copies artifacts back, and destroys the instance — all in one foreground command.

## Why
Manually running `vastai search offers ... | vastai create instance ... | vastai copy ... | ssh ... | vastai destroy ...` for every training run is tedious and error-prone (forgetting to destroy = ongoing GPU charges). `vastxm launch` does it in one line.

## Install

vastXM is a uv-managed Python package. The `vastai` CLI is a runtime requirement.

```bash
# 1. Install the vast.ai CLI globally and authenticate
pip install vastai
vastai set api-key <your_key_from_https://cloud.vast.ai/manage-keys/>

# 2. Install vastxm (editable, from the repo)
cd vastXM
uv tool install --editable .
```

After install, `vastxm` should be on your PATH.

## Quickstart

From a project directory:

```bash
vastxm launch --gpu A100 --num-gpus 4 --max-price 1.80 --disk 100 \
              --cmd 'deepspeed --num_gpus=4 train.py --config configs/nanochat_a100.yaml'
```

What happens:
1. Bundles the project as `.vastxm/bundle.tar.gz` (excludes `.git/`, `checkpoints/`, `logs/`, `*.pt`, etc.).
2. Picks the cheapest matching offer.
3. Creates the instance, waits for `running` (uses a public `pytorch:2.5.1-cuda12.4` image; installs `uv` in the onstart).
4. `vastai copy`s the bundle, then over SSH does `tar xzf`, `uv sync`, then runs your `--cmd`.
5. Streams output to your terminal *and* `./runs/<name>/train.log`.
6. On exit (success, failure, or Ctrl+C): pulls `/workspace/<bundle_dirname>/output/` back to `./runs/<name>/`, then destroys the instance.

Pass `--keep` to skip the destroy.

## `vastxm.toml` (project defaults)

Drop a `vastxm.toml` next to your training code so you don't have to repeat flags:

```toml
[defaults]
gpu = "A100"
num_gpus = 4
max_price = 1.80
disk = 100
image = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
bundle_root = "."
exclude = ["checkpoints/", "logs/", "data/"]
artifact_path = "/workspace/output"
artifact_dest = "./runs"
```

CLI flags override the toml; the toml overrides built-in defaults.

## CLI reference

| Command | What it does |
|---------|--------------|
| `vastxm launch --cmd '...' [flags]` | Full provision → run → destroy cycle. |
| `vastxm ls` | List your active vast.ai instances. |
| `vastxm logs <id>` | `tail -F` the remote `train.log`. |
| `vastxm ssh <id>` | Open an interactive shell on the instance. |
| `vastxm stop <id>` | Destroy an instance (alias for `vastai destroy instance`). |
| `vastxm pull <id> <remote> <local>` | Run `vastai copy` to fetch files. |

`vastxm launch` flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--cmd` | (required) | The full command to run on the instance, e.g. `'deepspeed --num_gpus=4 train.py --config x.yaml'`. |
| `--gpu` | `A100` | vast `gpu_name` filter. Examples: `RTX_4090`, `H100`, `A40`. |
| `--num-gpus` | `1` | |
| `--max-price` | `2.00` | USD/hr cap. |
| `--disk` | `50` | GB. |
| `--image` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | Any public Docker image. |
| `--bundle-root` | `.` | Directory tar'd into the bundle. |
| `--exclude PAT` | (defaults) | Repeatable. **Replaces** the default exclude list when used; include defaults explicitly if you still want them. |
| `--artifact-path` | `/workspace/output` | Remote dir copied back. |
| `--artifact-dest` | `./runs` | Local dir for artifact + log output. |
| `--name` | auto | Run label; controls the local run subdirectory. |
| `--keep` | off | Skip destroy on exit. |
| `--dry-run` | off | Print the plan; don't touch vastai. |
| `--config-file` | `vastxm.toml` | Path to project defaults file. |

## Troubleshooting

- **`The vastai CLI was not found on PATH`** → `pip install vastai` and confirm `which vastai`.
- **`No offers matched`** → raise `--max-price` or relax `--gpu`/`--num-gpus`. Run `vastai search offers '...'` directly to see what's available.
- **`instance ... reached terminal status 'exited'`** → the host failed to start the container; usually a transient host issue. Re-run `vastxm launch`. Use `VASTXM_DEBUG=1` for raw vastai output.
- **Stuck at `loading` for >15 min** → the image pull is too slow; pick a smaller image or a host with better bandwidth.
- **Lost track of an instance** → `vastxm ls` then `vastxm stop <id>`.
- **Want to debug remotely** → run with `--keep`, then `vastxm ssh <id>`.

## Limitations (by design)

- Single instance per launch — no multi-machine fan-out.
- No Docker registry build/push step. We pull a public image and `uv sync` at startup.
- No retry on host preemption — write your training to checkpoint regularly and re-run.
- Bundle excludes are simple glob patterns; we do not parse `.gitignore`. Use `--exclude` or `vastxm.toml` to keep the bundle small.

## Project layout

```
vastXM/
├── pyproject.toml
├── README.md                       ← you are here
├── src/vastxm/
│   ├── cli.py                      # argparse subcommands
│   ├── config.py                   # LaunchConfig + vastxm.toml loader
│   ├── vast.py                     # vastai CLI subprocess wrapper
│   ├── bundle.py                   # tar.gz creation
│   ├── instance.py                 # offer search, wait_for_running
│   ├── ssh.py                      # SSH stream helper
│   ├── workflow.py                 # launch orchestration
│   └── _log.py
└── tests/
```
