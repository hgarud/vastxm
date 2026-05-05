# vastXM

Run a training job on a rented [vast.ai](https://vast.ai) GPU with one command. vastXM rents a GPU, ships your code to it, runs your command, copies the results back, and shuts the instance down so you stop paying.

## Install

```bash
uv tool install --editable .
```

## Authenticate

Get your API key from <https://cloud.vast.ai/manage-keys/>, then:

```bash
vastxm auth <YOUR_API_KEY>
```

## Launch a job

From your project directory:

```bash
vastxm launch --gpu A100 --cmd 'python train.py'
```

vastXM will:

1. Show you a list of available GPUs — pick one with **↑/↓ + Enter** (Enter alone takes the cheapest).
2. Rent it, ship your code, run your command.
3. Copy `/workspace/output/` back into `./runs/<run-name>/`.
4. Destroy the instance.

Add `--keep` if you want to leave the instance running afterwards.

## Project defaults

Drop a `vastxm.toml` in your project so you don't repeat flags:

```toml
[defaults]
gpu = "A100"
num_gpus = 1
max_price = 1.80     # USD/hr ceiling
disk = 100           # GB
```

## Common GPU names

| Type | `--gpu` |
|---|---|
| A100 (any variant) | `A100` |
| H100 (any variant) | `H100` |
| H200 | `H200` |
| RTX 5090 | `RTX_5090` |
| RTX 4090 | `RTX_4090` |
| L40S | `L40S` |

## Other commands

```bash
vastxm ls              # list your running instances
vastxm ssh <id>        # open a shell on an instance
vastxm logs <id>       # tail the running command's log
vastxm stop <id>       # destroy an instance
```

## Tips

- **Don't forget to stop instances.** vast.ai charges by the hour. `vastxm ls` to see what's still running, `vastxm stop <id>` to kill it.
- **No offers found?** Raise `--max-price` or try a more common GPU.
- **Flag reference:** `vastxm launch --help`.
