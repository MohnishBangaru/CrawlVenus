MOCK_DATA = {
    "name": "TestUser",
    "first name": "TestUser",
    "full name": "TestUser",
    "username": "TestUser",
    "mobile": "9999999999",
    "phone": "9999999999",
    "telephone": "9999999999",
    "location": "Bangalore",
    "city": "Bangalore",
    "place": "Bangalore",
    "password": "Pass@123",
    "passcode": "Pass@123",
    "email": "test@example.com"
}


def get_mock_for(label: str) -> str:
    """Return mock value matching the semantic *label* of a text field.

    Parameters
    ----------
    label : str
        OCR-detected label/placeholder associated with an input field.
    """
    key = label.lower()
    for k, v in MOCK_DATA.items():
        if k in key:
            return v
    return "test"


def autofill_if_keyboard_visible(is_keyboard_visible_func) -> None:  # pragma: no cover
    """Convenience helper. If the soft-keyboard is open, insert a suitable mock value.

    Pass the Vision/ADB utility `is_keyboard_visible` so this module stays framework-agnostic.
    """
    import subprocess, random, time
    # Debounce: fill only once per keyboard visibility session
    if not is_keyboard_visible_func():
        # Reset flag when keyboard hidden
        autofill_if_keyboard_visible._filled = False  # type: ignore
        return

    if getattr(autofill_if_keyboard_visible, "_filled", False):  # type: ignore
        # Extra guard: avoid rapid re-fill if keyboard visibility flickers
        import time
        last_ts = getattr(autofill_if_keyboard_visible, "_last_ts", 0)  # type: ignore
        if time.time() - last_ts < 3:  # 3-second quiet window
            return

    # Cycle through deterministic but varied keys for successive fills
    order = ["name", "mobile", "email", "location", "password"]
    autofill_if_keyboard_visible._idx = getattr(autofill_if_keyboard_visible, "_idx", 0) + 1  # type: ignore
    label = order[autofill_if_keyboard_visible._idx % len(order)]
    value = MOCK_DATA[label]

    escaped = value.replace(" ", "%s")
    subprocess.run(["adb", "shell", "input", "text", escaped], capture_output=True, check=False)
    time.sleep(0.2)

    # record time
    import time as _t
    autofill_if_keyboard_visible._last_ts = _t.time()  # type: ignore

    # mark filled
    autofill_if_keyboard_visible._filled = True  # type: ignore

# ------------------
# Background watcher to call autofill periodically without explicit tester hook
# ------------------


def _is_keyboard_visible() -> bool:
    """Return True if soft keyboard currently shown (uses adb dumpsys)."""
    result = subprocess.run([
        "adb",
        "shell",
        "dumpsys",
        "input_method"
    ], capture_output=True, text=True, check=False)
    return ("mInputShown=true" in result.stdout) or ("mIsInputViewShown=true" in result.stdout)


def _watcher() -> None:  # pragma: no cover
    while True:
        try:
            autofill_if_keyboard_visible(_is_keyboard_visible)
        except Exception:
            pass
        time.sleep(0.7)


# start watcher thread once on import
import threading as _th

_t = _th.Thread(target=_watcher, daemon=True)
_t.start() 