import subprocess


def test_uv_lock_check() -> None:
    subprocess.run(["uv", "lock", "--check"], check=True)
