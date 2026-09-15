"""Secret lookup across environment variables, Kaggle and Colab."""

from __future__ import annotations

import os


def get_secret(name: str) -> str:
    """Return a secret from the environment, Kaggle Secrets or Colab userdata.

    Raises:
        RuntimeError: if the secret is not found anywhere.
    """
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - not on Kaggle, or the secret is not attached
        pass
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:  # noqa: BLE001 - not on Colab, or access not granted
        pass
    raise RuntimeError(
        f"{name} not found. Set it as an env var, add it under Kaggle "
        f"Add-ons > Secrets (tick the attach box), or use Colab's key sidebar."
    )
