"""Hugging Face account: which token the engine uses, and whether it can open a gated model.

The desktop shell keeps the token in the OS keychain and starts the engine with HF_TOKEN; setting a new
one here changes it in the running engine (and in the training and export workers it starts) without a
restart. A token from `huggingface-cli login` is used when Temper has none of its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi, constants, get_token
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from pydantic import BaseModel

router = APIRouter(prefix="/api/hf", tags=["account"])


def _source() -> str | None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "temper"
    if Path(constants.HF_TOKEN_PATH).is_file():
        return "cli"
    return None


def _whoami(token: str) -> dict:
    info = HfApi().whoami(token=token)
    return {"name": info.get("name"), "fullname": info.get("fullname"), "orgs": [o.get("name") for o in info.get("orgs", [])]}


@router.get("/account")
def account():
    token = get_token()
    if not token:
        return {"signed_in": False, "source": None}
    try:
        return {"signed_in": True, "source": _source(), **_whoami(token)}
    except HfHubHTTPError:
        return {"signed_in": False, "source": _source(), "invalid": True}
    except OSError:  # offline: the token may be fine
        return {"signed_in": True, "source": _source(), "name": None, "offline": True}


class TokenRequest(BaseModel):
    token: str


@router.post("/token")
def set_token(req: TokenRequest):
    token = req.token.strip()
    if not token.startswith("hf_"):
        raise HTTPException(400, "That doesn't look like a Hugging Face token. They start with hf_.")
    try:
        who = _whoami(token)
    except HfHubHTTPError as e:
        raise HTTPException(400, "Hugging Face didn't accept this token. Check that you copied all of it, or create a new one.") from e
    except OSError as e:
        raise HTTPException(503, "Temper couldn't reach Hugging Face to check the token. Try again when you're online.") from e
    os.environ["HF_TOKEN"] = token
    return {"signed_in": True, "source": "temper", **who}


@router.delete("/token")
def delete_token():
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return account()


@router.get("/access")
def access(repo_id: str):
    """Whether this machine's token can download `repo_id`, and what to do if not."""
    url = f"https://huggingface.co/{repo_id}"
    has_token = bool(get_token())
    try:
        HfApi().auth_check(repo_id)
        return {"access": True, "gated": False, "has_token": has_token, "url": url}
    except GatedRepoError:
        if not has_token:
            message = "This model asks you to accept its licence first. Accept it on the model's page, then add a Hugging Face token in Settings."
        else:
            message = "Your account hasn't been given access yet. Accept the licence on the model's page; some publishers review requests, which can take a while."
        return {"access": False, "gated": True, "has_token": has_token, "url": url, "message": message}
    except RepositoryNotFoundError:
        message = "This model is private or doesn't exist." + ("" if has_token else " If it's yours, add a Hugging Face token in Settings.")
        return {"access": False, "gated": False, "has_token": has_token, "url": url, "message": message}
