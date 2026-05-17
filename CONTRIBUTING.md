# Contributing to ExitKit

Thanks for your interest in ExitKit.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

## Pull requests

- One conceptual change per PR.
- Add tests that fail before your change and pass after.
- Run `ruff check .` and `mypy` locally; both must pass.
- Keep the public API surface tight — v0.1 deliberately exposes only the continuity metric.

## Scope guard

ExitKit v0.1 is the single *continuer-select* component. Proposals to bundle a UI, a policy engine, or a provenance store are appreciated but will be staged for v0.2+ rather than merged into v0.1.

## License

By submitting a contribution you agree it will be licensed under the project's [Apache 2.0 License](LICENSE).
