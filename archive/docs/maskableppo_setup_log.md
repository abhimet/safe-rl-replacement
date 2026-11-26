# Published MaskablePPO reference

**Feasibility decision — NO-GO in this workspace.** The matching publication is Pranger and Könighofer's *tempestpy: Easy-to-Use Shielded Reinforcement Learning*. Section 5 uses Stable-Baselines3 PPO and MaskablePPO on slippery-cliff and adversarial MiniGridSafe tasks, reporting averages over five runs for each condition. The example shows 100,000 training timesteps. The workflow generates a PRISM model and applies action masks from a model-checked safety property. The local project instead uses Double DQN, deterministic coordinate GridWorld observations, immediate wall and obstacle checks, and post-proposal replacement. The setups yield different outcomes and support a methodological comparison only. [Publication](https://link.springer.com/chapter/10.1007/978-3-032-40121-2_9)

Official sources identified were [tempestpy](https://github.com/isec-tugraz/tempestpy), [MinigridSafe](https://github.com/isec-tugraz/MinigridSafe), and [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib). Official installation requires the native Tempest/Storm build dependencies. [Installation guide](https://tempest-synthesis.org/installation/)

On Windows with Python 3.14.0, both Git HEAD lookups and the pip dependency attempt failed during source fetch with `SSL certificate problem: unable to get local issuer certificate`. A Git retry using Windows Schannel returned `SEC_E_NO_CREDENTIALS`. The local Docker CLI could not connect to the Docker engine. No official training or evaluation ran, and no installed source revision or resolved benchmark configuration was obtained.

```powershell
git ls-remote https://github.com/isec-tugraz/tempestpy.git HEAD
git ls-remote https://github.com/isec-tugraz/MinigridSafe.git HEAD
git -c http.sslBackend=schannel ls-remote https://github.com/isec-tugraz/tempestpy.git HEAD
git -c http.sslBackend=schannel ls-remote https://github.com/isec-tugraz/MinigridSafe.git HEAD
docker info --format '{{.ServerVersion}}'
.\.venv\Scripts\python.exe -m pip --disable-pip-version-check install --target .venv/benchmark-deps --no-cache-dir --retries 0 --timeout 10 'git+https://github.com/isec-tugraz/tempestpy.git' 'git+https://github.com/isec-tugraz/MinigridSafe.git' sb3-contrib
```


## Follow-up environment checks

- `wsl.exe --list --verbose` returned `Wsl/E_ACCESSDENIED`.
- A HEAD request to the official tempestpy source archive at `https://codeload.github.com/isec-tugraz/tempestpy/zip/refs/heads/main` failed during TLS negotiation with Schannel error `SEC_E_NO_CREDENTIALS (0x8009030e)`.
- The Python launcher detected Python 3.10 and 3.14. Neither source retrieval nor native build tooling was available through the tested routes. CMake, Ninja, and the Visual C++ compiler were absent from PATH.

The feasibility decision remains **NO-GO in this workspace**. The reference still requires a working source retrieval route and the official Tempest build environment before training can begin.
