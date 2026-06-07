"""Load .env from project root (optional — no-op if python-dotenv missing)."""
import pathlib


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = pathlib.Path(__file__).resolve().parent
    load_dotenv(root / ".env")
