"""
test_api.py – Automated test suite for the Face Mask Detection API

Usage:
    # Make sure the API is running first:
    #   uvicorn app:app --port 8000
    #   or: docker run -p 8000:8000 mask-detection-api

    python test_api.py                        # default: http://localhost:8000
    python test_api.py --base-url http://... # custom host
    python test_api.py --image path/to/face.jpg   # quick single-image test
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Optional: coloured output ────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = RESET = ""

# ── Optional: requests (falls back to urllib) ─────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Optional: numpy / PIL for synthetic image generation ─────────────────────
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


BASE_URL = "http://localhost:8000"
TIMEOUT  = 30  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print(label: str, msg: str, color: str = ""):
    print(f"{color}[{label}]{RESET} {msg}")


def pass_(msg):  _print("PASS", msg, GREEN)
def fail_(msg):  _print("FAIL", msg, RED)
def info_(msg):  _print("INFO", msg, CYAN)
def warn_(msg):  _print("WARN", msg, YELLOW)


def get_json(url: str) -> dict:
    """Simple GET → JSON via urllib (no extra deps required)."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def post_image_urllib(url: str, img_bytes: bytes, filename: str = "test.jpg") -> dict:
    """Multipart POST using only stdlib urllib."""
    boundary = "----TestBoundary1234567890"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def post_image_requests(url: str, img_bytes: bytes, filename: str = "test.jpg") -> dict:
    resp = requests.post(url, files={"file": (filename, img_bytes, "image/jpeg")}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def post_image(url: str, img_bytes: bytes, filename: str = "test.jpg") -> dict:
    if HAS_REQUESTS:
        return post_image_requests(url, img_bytes, filename)
    return post_image_urllib(url, img_bytes, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test images (no real photos needed)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_face_jpg(width=224, height=224, skin_rgb=(210, 170, 130)) -> bytes:
    """Create a simple synthetic face-like JPEG for smoke testing."""
    if not HAS_PIL:
        # Fallback: minimal valid 1×1 JPEG
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xb1"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
            b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
            b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81"
            b"\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19"
            b"\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86"
            b"\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4"
            b"\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2"
            b"\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9"
            b"\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
            b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4"
            b"\xff\xd9"
        )
    img = Image.new("RGB", (width, height), color=skin_rgb)
    draw = ImageDraw.Draw(img)
    # rough face oval
    draw.ellipse([40, 30, 184, 210], fill=skin_rgb, outline=(150, 120, 90), width=2)
    # eyes
    draw.ellipse([70, 90, 95, 110],  fill=(50, 30, 20))
    draw.ellipse([129, 90, 154, 110], fill=(50, 30, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _solid_color_jpg(color=(128, 128, 128)) -> bytes:
    """Tiny solid-colour JPEG for edge-case tests."""
    if not HAS_PIL:
        return _synthetic_face_jpg()
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes() -> bytes:
    if not HAS_PIL:
        # 1×1 red PNG
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8"
            b"\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    img = Image.new("RGB", (32, 32), color=(200, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Individual tests
# ─────────────────────────────────────────────────────────────────────────────

results = []


def run(name: str, fn):
    info_(f"Running: {name}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        pass_(f"{name}  ({elapsed*1000:.0f} ms)")
        results.append((name, True, None))
    except AssertionError as e:
        fail_(f"{name}  →  {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        fail_(f"{name}  →  {type(e).__name__}: {e}")
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ── 1. Health ────────────────────────────────────────────────────────────────
def test_root():
    data = get_json(f"{BASE_URL}/")
    assert "message" in data, f"Expected 'message' key, got {data}"


def test_health():
    data = get_json(f"{BASE_URL}/health")
    assert data.get("status") == "ok",        f"status != ok: {data}"
    assert data.get("model_loaded") is True,  f"model not loaded: {data}"
    assert "device"  in data,                 f"missing 'device': {data}"
    assert "classes" in data,                 f"missing 'classes': {data}"


# ── 2. Predict – valid images ─────────────────────────────────────────────────
def test_predict_jpeg():
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img, "face.jpg")
    _assert_valid_prediction(data)


def test_predict_png():
    img = _png_bytes()
    data = post_image(f"{BASE_URL}/predict", img, "face.png")
    _assert_valid_prediction(data)


def test_predict_response_fields():
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    required = {"status", "class", "confidence", "action", "all_probabilities"}
    missing = required - data.keys()
    assert not missing, f"Missing fields: {missing}"


def test_predict_confidence_range():
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    c = data["confidence"]
    assert 0.0 <= c <= 1.0, f"Confidence out of range: {c}"


def test_predict_valid_class():
    valid_classes = {"WithMask", "WithoutMask"}
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    assert data["class"] in valid_classes, f"Unknown class: {data['class']}"


def test_predict_valid_status():
    valid_statuses = {"mask_on", "mask_off"}
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    assert data["status"] in valid_statuses, f"Unknown status: {data['status']}"


def test_predict_valid_action():
    valid_actions = {"Allow entry", "Deny entry"}
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    assert data["action"] in valid_actions, f"Unknown action: {data['action']}"


def test_predict_all_probabilities_sum():
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    probs = data.get("all_probabilities", {})
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to ~1: {total}"


def test_predict_status_action_consistency():
    """If status is mask_on, action must be Allow entry, and vice-versa."""
    img = _synthetic_face_jpg()
    data = post_image(f"{BASE_URL}/predict", img)
    mapping = {"mask_on": "Allow entry", "mask_off": "Deny entry"}
    expected_action = mapping.get(data["status"])
    assert data["action"] == expected_action, (
        f"Inconsistent status/action: {data['status']} → {data['action']}"
    )


# ── 3. Different image sizes ──────────────────────────────────────────────────
def test_small_image():
    img = _solid_color_jpg()
    data = post_image(f"{BASE_URL}/predict", img, "small.jpg")
    _assert_valid_prediction(data)


def test_large_image():
    if not HAS_PIL:
        warn_("PIL not available – skipping large image test")
        return
    large = Image.new("RGB", (1920, 1080), color=(180, 140, 100))
    buf = io.BytesIO()
    large.save(buf, format="JPEG")
    data = post_image(f"{BASE_URL}/predict", buf.getvalue(), "large.jpg")
    _assert_valid_prediction(data)


# ── 4. Error handling ─────────────────────────────────────────────────────────
def test_missing_file():
    """POST with no file should return 422."""
    url = f"{BASE_URL}/predict"
    if HAS_REQUESTS:
        resp = requests.post(url, timeout=TIMEOUT)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    else:
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            urllib.request.urlopen(req, timeout=TIMEOUT)
            assert False, "Expected HTTP error, got 200"
        except urllib.error.HTTPError as e:
            assert e.code == 422, f"Expected 422, got {e.code}"


def test_invalid_file_type():
    """Upload a text file – should return 415."""
    url = f"{BASE_URL}/predict"
    txt = b"this is not an image"
    boundary = "----Bound"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode() + txt + f"\r\n--{boundary}--\r\n".encode()

    if HAS_REQUESTS:
        resp = requests.post(url, files={"file": ("test.txt", txt, "text/plain")}, timeout=TIMEOUT)
        assert resp.status_code == 415, f"Expected 415, got {resp.status_code}"
    else:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=TIMEOUT)
            assert False, "Expected HTTP error for invalid type"
        except urllib.error.HTTPError as e:
            assert e.code == 415, f"Expected 415, got {e.code}"


# ── 5. Real image file test ───────────────────────────────────────────────────
def test_real_image(path: str):
    p = Path(path)
    assert p.exists(), f"File not found: {path}"
    img_bytes = p.read_bytes()
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    if HAS_REQUESTS:
        resp = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (p.name, img_bytes, mime)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    else:
        data = post_image(f"{BASE_URL}/predict", img_bytes, p.name)

    _assert_valid_prediction(data)
    print(f"         Result → class={data['class']}  status={data['status']}  "
          f"confidence={data['confidence']:.3f}  action={data['action']}")


# ── 6. Load / latency ─────────────────────────────────────────────────────────
def test_latency():
    img = _synthetic_face_jpg()
    times = []
    for _ in range(5):
        t0 = time.time()
        post_image(f"{BASE_URL}/predict", img)
        times.append(time.time() - t0)
    avg_ms = sum(times) / len(times) * 1000
    info_(f"Average latency over 5 requests: {avg_ms:.0f} ms")
    assert avg_ms < 5000, f"Average latency too high: {avg_ms:.0f} ms"


# ── Helper ────────────────────────────────────────────────────────────────────
def _assert_valid_prediction(data: dict):
    assert "class"      in data, f"Missing 'class': {data}"
    assert "confidence" in data, f"Missing 'confidence': {data}"
    assert "status"     in data, f"Missing 'status': {data}"
    assert "action"     in data, f"Missing 'action': {data}"
    assert data["class"] in {"WithMask", "WithoutMask"}, f"Bad class: {data['class']}"
    assert 0.0 <= data["confidence"] <= 1.0,             f"Bad confidence: {data['confidence']}"


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_api(timeout: int = 30):
    info_(f"Waiting for API at {BASE_URL} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get_json(f"{BASE_URL}/health")
            info_("API is up!")
            return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"API did not become ready within {timeout}s")


def main():
    global BASE_URL

    parser = argparse.ArgumentParser(description="Face Mask Detection API test suite")
    parser.add_argument("--base-url", default=BASE_URL, help="API base URL")
    parser.add_argument("--image",    default=None,     help="Path to a real image for an extra test")
    parser.add_argument("--wait",     action="store_true", help="Wait for API to become ready")
    args = parser.parse_args()

    BASE_URL = args.base_url.rstrip("/")

    if args.wait:
        wait_for_api()

    print("\n" + "═" * 60)
    print("  Face Mask Detection API – Test Suite")
    print("  Target:", BASE_URL)
    print("═" * 60 + "\n")

    # Core tests
    run("GET  /",                          test_root)
    run("GET  /health",                    test_health)
    run("POST /predict – JPEG",            test_predict_jpeg)
    run("POST /predict – PNG",             test_predict_png)
    run("POST /predict – response fields", test_predict_response_fields)
    run("POST /predict – confidence range",test_predict_confidence_range)
    run("POST /predict – valid class",     test_predict_valid_class)
    run("POST /predict – valid status",    test_predict_valid_status)
    run("POST /predict – valid action",    test_predict_valid_action)
    run("POST /predict – probs sum to 1",  test_predict_all_probabilities_sum)
    run("POST /predict – status↔action",  test_predict_status_action_consistency)
    run("POST /predict – small image",     test_small_image)
    run("POST /predict – large image",     test_large_image)
    run("POST /predict – missing file",    test_missing_file)
    run("POST /predict – invalid type",    test_invalid_file_type)
    run("POST /predict – latency",         test_latency)

    # Optional real-image test
    if args.image:
        run(f"POST /predict – real image ({args.image})",
            lambda: test_real_image(args.image))

    # ── Summary ──────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print("\n" + "═" * 60)
    print(f"  Results: {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}  "
          f"|  {len(results)} total")
    print("═" * 60 + "\n")

    if failed:
        print(f"{RED}Failed tests:{RESET}")
        for name, ok, err in results:
            if not ok:
                print(f"  • {name}: {err}")
        sys.exit(1)

    print(f"{GREEN}All tests passed! ✅{RESET}\n")


if __name__ == "__main__":
    main()
