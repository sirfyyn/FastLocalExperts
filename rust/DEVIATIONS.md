# Deliberate differences from the Python Studio

The Python FLE Studio is the specification of `fle-api`, and the tests compare both request by request. Where Python has a
mistake, this server does **not** copy it: the difference is listed here, and the test knows about it (it converts the Python
text or skips the Python call, and asserts the intended behaviour on the Rust side instead).

| Id | Python | Here | Why |
|---|---|---|---|
| D2 | A start next to an attached server is allowed and runs a second engine | 409 "A server is attached. Disconnect it on the Engine page first." | Two engines at once eat the RAM and GPU memory of one |
| D3 | Removing an engine whose data folder is behind a symlink does not delete the folder (resolved and unresolved paths are compared) | Both sides resolved: the folder is deleted | Plain bug; the tests run without symlinks in the path so the two agree |
| D4 | `/api/tools` `servers` are real MCP connections | Listed, status "error: not supported by this server yet" | MCP is not ported yet |
| D5 | Unported routes answer 404 | 501 "<feature> is not in this version of FLE yet." | A clear message, not something that looks like a bug |
| D7 | `/api/chat/capabilities` `pdf` is true when `pypdf` is installed | always false | no PDF text extraction yet |
| D8 | The page's `system.python` is the Python version | empty string | this server is not Python |
| D9 | A pure-Python crash for `{"results": 5}` from a search server | an empty result | robustness |

Gone from the list because Python fixed them itself (0.4.1): D1, a plan error with a hint was printed as a tuple; now the two
sentences are joined, as here. D6 (the update panel) and the hosts list are plugins of the Python Studio now; the core, which is
what this server replaces, has neither.

**Which Python.** The specification is the *core* of FLE Studio `main` at the commit in `STUDIO_PIN`, started with
`FLE_STUDIO_NO_BUNDLED=1` (no plugins: Forge, Updates, Remote access, Recorder and the model switcher are Python plugins there).
Moving the pin is a deliberate act: change the file, run the comparison tests, port what differs.

Things that stay the same on purpose even though they look odd (and are candidates for a later, announced change): a start without a
model name uses the last model; chat lists with equal times keep the file system's order; a page is sent with `Cache-Control` twice.
