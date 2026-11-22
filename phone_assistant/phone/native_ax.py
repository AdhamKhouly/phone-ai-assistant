"""macOS Accessibility driver for the native "DIGITS from T-Mobile" app (bundle com.tmobile.phone2).

This is the ONLY module that talks to macOS. It uses the public Accessibility API (the same one
VoiceOver uses) and nothing else:
  * read():   returns the app's windows as plain dict trees (roles, labels, DOM ids, text).
  * press():  performs the standard AXPress action on ONE element handed to it.
  * quit() / launch(): ordinary app lifecycle (like Cmd-Q / opening the app).
  * hide() / frontmost_pid() / activate_pid(): keep DIGITS in the background (see "Staying in
    the background" below).

It never sends keystrokes or mouse events, never reads the app's files, cookies or network
traffic, and never touches account settings. The app is an Electron app, so its web content is
only exposed after the documented `AXManualAccessibility` attribute is set on the app element
(https://www.electronjs.org/docs/latest/tutorial/accessibility); that flag resets on app restart.

macOS permission needed: Accessibility, for the program that runs the bot (Terminal / Claude).

Staying in the background (all measured on this Mac, 2026-11-21):
  * A HIDDEN DIGITS app (as after Cmd-H) is fully readable: its window is listed and its content
    can be read. Hiding it never changes which app is in front.
  * If you close the DIGITS window with the red button, the window disappears from the readable
    tree. Hiding the app brings it back into the tree without showing anything.
  * Sending DIGITS an "open" request while it is already running makes DIGITS bring ITSELF to the
    front, even when the request says not to. So the bot never does that for a merely closed window.
  * A freshly started DIGITS also puts itself in front, and it must be shown once before its
    content becomes readable. The bot hides it as soon as it is readable (about half a second).
"""
from __future__ import annotations

import logging
import subprocess
import time

log = logging.getLogger(__name__)

BUNDLE_ID = "com.tmobile.phone2"
MAX_NODES = 3000
MAX_DEPTH = 45
_ATTRS = ["AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXDOMIdentifier", "AXValue", "AXChildren"]


def _s(v) -> str:
    return str(v) if isinstance(v, str) else ""


class MacAXSource:
    """Reads the DIGITS app's accessibility tree. One instance per bot process."""

    def __init__(self, bundle_id: str = BUNDLE_ID) -> None:
        import ApplicationServices as AS  # pyobjc; imported lazily so tests don't need it
        import AppKit

        self._AS, self._AppKit = AS, AppKit
        self.bundle_id = bundle_id
        self._pid: int | None = None
        self._app = None
        self._enabled_at = 0.0
        self.last_window_count = 0

    # ---- status -----------------------------------------------------------------------
    def trusted(self) -> bool:
        """True if macOS has granted this process Accessibility access."""
        return bool(self._AS.AXIsProcessTrusted())

    def _running_app(self):
        apps = self._AppKit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(self.bundle_id)
        return apps[0] if apps else None

    def pid(self) -> int | None:
        app = self._running_app()
        return int(app.processIdentifier()) if app is not None else None

    # ---- read (no side effects on calls) -------------------------------------------------
    def read(self) -> list[dict] | None:
        """Window trees, or None when the app is not running."""
        AS = self._AS
        pid = self.pid()
        if pid is None:
            self._pid = self._app = None
            return None
        if pid != self._pid:
            self._pid, self._app, self._enabled_at = pid, AS.AXUIElementCreateApplication(pid), 0.0
            AS.AXUIElementSetMessagingTimeout(self._app, 1.0)
        err, wins = AS.AXUIElementCopyAttributeValue(self._app, "AXWindows", None)
        if err != 0 or wins is None:
            self.last_window_count = 0
            return []
        self.last_window_count = len(wins)
        budget = [MAX_NODES]
        trees = [self._node(w, 0, budget) for w in wins]
        has_web = any(n["role"] == "AXWebArea" for t in trees for n in iter_nodes(t))
        if not has_web and time.monotonic() - self._enabled_at > 5.0:
            # Ask Electron to expose its web content to assistive clients (resets on restart).
            AS.AXUIElementSetAttributeValue(self._app, "AXManualAccessibility", True)
            self._enabled_at = time.monotonic()
        return trees

    def _node(self, el, depth: int, budget: list[int]) -> dict:
        budget[0] -= 1
        err, vals = self._AS.AXUIElementCopyMultipleAttributeValues(el, _ATTRS, 0, None)
        if err != 0 or vals is None:
            return {"role": "", "subrole": "", "title": "", "desc": "", "id": "", "value": "", "children": [], "_el": el}
        kids = vals[6]
        children: list[dict] = []
        if depth < MAX_DEPTH and kids is not None and not isinstance(kids, str) and hasattr(kids, "__iter__"):
            try:
                for c in kids:
                    if budget[0] <= 0:
                        break
                    children.append(self._node(c, depth + 1, budget))
            except TypeError:
                pass
        return {"role": _s(vals[0]), "subrole": _s(vals[1]), "title": _s(vals[2]), "desc": _s(vals[3]),
                "id": _s(vals[4]), "value": _s(vals[5]), "children": children, "_el": el}

    # ---- the single action primitive ---------------------------------------------------
    def press(self, node: dict) -> bool:
        """AXPress on one element. Callers (answer / hang up only) decide which element."""
        err = self._AS.AXUIElementPerformAction(node["_el"], "AXPress")
        return err == 0

    def open_tel(self, number: str) -> bool:
        """Hand DIGITS a standard tel: link, which it is registered to handle. This is the only way
        the assistant starts a call. `number` must be +digits; anything else is refused."""
        if not (number.startswith("+") and number[1:].isdigit() and 8 <= len(number) - 1 <= 15):
            return False
        try:
            ws = self._AppKit.NSWorkspace.sharedWorkspace()
            app = ws.URLForApplicationWithBundleIdentifier_(self.bundle_id)
            url = self._AppKit.NSURL.URLWithString_("tel:" + number)
            if app is None or url is None:
                return False
            cfg = self._AppKit.NSWorkspaceOpenConfiguration.configuration()
            cfg.setActivates_(False)
            cfg.setAddsToRecentItems_(False)
            ws.openURLs_withApplicationAtURL_configuration_completionHandler_([url], app, cfg, None)
            return True
        except Exception as e:
            log.warning("could not hand the number to DIGITS (%s)", type(e).__name__)
            return False

    # ---- app lifecycle ------------------------------------------------------------------
    def quit(self, wait_s: float = 5.0) -> bool:
        """Quit the app (this drops any call in progress). Force-quit only if a normal quit fails."""
        app = self._running_app()
        if app is None:
            return True
        app.terminate()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self._running_app() is None:
                return True
            time.sleep(0.2)
        log.warning("DIGITS did not quit normally; force-quitting")
        app = self._running_app()
        if app is not None:
            app.forceTerminate()
            time.sleep(1.0)
        return self._running_app() is None

    def launch(self) -> bool:
        """Start the app (or ask a running one to show its window) with activation turned off.
        DIGITS may still put itself in front; callers hide it again once it is readable."""
        try:
            ws = self._AppKit.NSWorkspace.sharedWorkspace()
            url = ws.URLForApplicationWithBundleIdentifier_(self.bundle_id)
            if url is not None:
                cfg = self._AppKit.NSWorkspaceOpenConfiguration.configuration()
                cfg.setActivates_(False)
                cfg.setAddsToRecentItems_(False)
                ws.openApplicationAtURL_configuration_completionHandler_(url, cfg, None)
                return True
        except Exception as e:
            log.warning("background launch through NSWorkspace failed (%s); using `open -g`", type(e).__name__)
        try:
            return subprocess.run(["/usr/bin/open", "-g", "-b", self.bundle_id], timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # ---- staying in the background --------------------------------------------------------
    def frontmost_pid(self) -> int | None:
        """Process id of the app that has keyboard focus. Only the id is read: no window titles,
        no document names, nothing typed."""
        AS = self._AS
        try:
            err, app = AS.AXUIElementCopyAttributeValue(AS.AXUIElementCreateSystemWide(), "AXFocusedApplication", None)
            if err == 0 and app is not None:
                err, pid = AS.AXUIElementGetPid(app, None)
                if err == 0 and pid:
                    return int(pid)
        except Exception:
            pass
        try:   # the call above sometimes answers nothing; fall back to the workspace's view
            app = self._AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is not None:
                return int(app.processIdentifier())
        except Exception:
            pass
        return None

    def is_frontmost(self) -> bool:
        """Is DIGITS the app in front? Asked of DIGITS itself, which answers reliably."""
        pid = self.pid()
        if pid is None:
            return False
        try:
            err, v = self._AS.AXUIElementCopyAttributeValue(self._AS.AXUIElementCreateApplication(pid), "AXFrontmost", None)
            if err == 0:
                return bool(v)
        except Exception:
            pass
        return self.frontmost_pid() == pid

    def is_hidden(self) -> bool:
        pid = self.pid()
        if pid is None:
            return False
        err, v = self._AS.AXUIElementCopyAttributeValue(self._AS.AXUIElementCreateApplication(pid), "AXHidden", None)
        return err == 0 and bool(v)

    def hide(self) -> bool:
        """Hide the app, exactly like Cmd-H. It keeps running and stays readable."""
        app = self._running_app()
        if app is None:
            return False
        app.hide()
        return True

    def activate_pid(self, pid: int) -> bool:
        """Give focus back to the app that had it (used once, after DIGITS pushed itself forward)."""
        try:
            app = self._AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is None or app.isTerminated():
                return False
            if app.bundleIdentifier() == self.bundle_id:
                return False                    # never used to bring DIGITS itself forward
            return bool(app.activateWithOptions_(0))
        except Exception:
            return False


def iter_nodes(node: dict):
    yield node
    for c in node.get("children", ()):
        yield from iter_nodes(c)
