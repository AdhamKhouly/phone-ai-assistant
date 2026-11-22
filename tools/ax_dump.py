"""Read-only dump of the DIGITS app's accessibility tree (roles + labels). Never performs actions.

Usage: .venv/bin/python tools/ax_dump.py [--enable] [--full] [--depth N]
  --enable  set Electron's documented AXManualAccessibility flag so web content is exposed
  --full    print text values too (default: only controls' labels; static text is length-only)
"""
import sys
from ApplicationServices import (AXIsProcessTrusted, AXUIElementCopyActionNames,
                                 AXUIElementCopyAttributeValue, AXUIElementCreateApplication,
                                 AXUIElementSetAttributeValue)
from AppKit import NSRunningApplication

BUNDLE = "com.tmobile.phone2"
full = "--full" in sys.argv
depth_max = int(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 40


def attr(el, name):
    err, val = AXUIElementCopyAttributeValue(el, name, None)
    return val if err == 0 else None


def walk(el, depth=0, counter=[0]):
    counter[0] += 1
    if depth > depth_max or counter[0] > 4000:
        return
    role = attr(el, "AXRole") or "?"
    bits = []
    for a in ("AXSubrole", "AXTitle", "AXDescription", "AXDOMIdentifier", "AXDOMClassList"):
        v = attr(el, a)
        if v:
            bits.append(f"{a[2:]}={str(list(v) if a == 'AXDOMClassList' else v)[:90]!r}")
    v = attr(el, "AXValue")
    if v is not None and str(v):
        s = str(v)
        interactive = role in ("AXButton", "AXLink", "AXCheckBox", "AXRadioButton", "AXTab", "AXMenuItem", "AXHeading")
        bits.append(f"Value={s[:90]!r}" if (full or interactive) else f"Value=<{len(s)} chars>")
    err, acts = AXUIElementCopyActionNames(el, None)
    if err == 0 and acts and role in ("AXButton", "AXLink", "AXCheckBox", "AXMenuItem", "AXTab", "AXRadioButton"):
        bits.append("actions=" + ",".join(a[2:] for a in acts))
    if role not in ("AXGroup",) or bits:
        print("  " * depth + role + " " + " ".join(bits))
    for c in attr(el, "AXChildren") or []:
        walk(c, depth + 1)


apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(BUNDLE)
print("trusted:", AXIsProcessTrusted(), "running:", len(apps))
if not apps:
    sys.exit(1)
app = AXUIElementCreateApplication(apps[0].processIdentifier())
print("AXManualAccessibility before:", attr(app, "AXManualAccessibility"))
if "--enable" in sys.argv:
    print("set ->", AXUIElementSetAttributeValue(app, "AXManualAccessibility", True))
    import time; time.sleep(1.5)
for w in attr(app, "AXWindows") or []:
    walk(w)
