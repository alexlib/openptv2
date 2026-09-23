# Browser GUI (Xpra) — the real desktop GUI, no install

Serves the existing `pyptv_gui` (TraitsUI/Chaco/PySide6, unmodified) to a
browser via [Xpra](https://xpra.org)'s HTML5 client. No new GUI code, no
Qt-to-WebAssembly port — the app runs exactly as it does on the desktop,
inside a virtual X display (`Xvfb`), and Xpra forwards its window to
whatever browser opens the URL.

Design background, and why this approach was chosen over a from-scratch
browser-native rewrite, is in the project's `design/browser-architecture.md`.
The short version: emscripten-forge's Qt-in-WASM pipeline cross-compiles
C++/CMake Qt Widgets projects, and openptv2's GUI is Python (no PySide6-for-
WASM exists anywhere), so recompiling it into the browser directly isn't
viable. Serving the real, working desktop app remotely sidesteps that
entirely. The pattern here follows
[CyVerse VICE](https://github.com/cyverse-vice/xpra), which serves other
scientific desktop GUI apps (CellProfiler, Orange3, ImageJ) to browsers the
same way.

## Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile.xpra` | Same GUI install as `docker/Dockerfile`, plus Xvfb + Xpra |
| `docker/entry-xpra.sh` | Starts Xpra with `openptv2-gui` as its child process |
| `docker/run-xpra.sh` | Local helper (Linux/macOS/WSL2/Git Bash): build once, run, open a browser tab |
| `docker/run-xpra.ps1` | Same, for native Windows PowerShell |

## Test it locally

```bash
docker build -f docker/Dockerfile.xpra -t openptv2-xpra .
./docker/run-xpra.sh
```

Then open **http://localhost:9876**. `pyptv_gui` should appear as a window
in the page — same menu bar, tree, and panels as the desktop version. The
baked `test_cavity` demo is copied to `/data` automatically if nothing else
is mounted there, so there's something to open immediately (`File → Open`
→ the mounted folder).

To point it at your own data instead of the baked demo:

```bash
./docker/run-xpra.sh /path/to/your/experiment
```

**What "working" looks like, concretely, before trusting this further:**
the window renders with menus/tree/panels intact, you can open an
experiment, run detection on a frame, and — the one real unknown — Chaco's
plots draw correctly (not blank or garbled). Chaco/Enable's Qt backend is
normally software-rendered (QPainter, not real GPU), so it's expected to
work under Xvfb with no GPU, but this hasn't been click-tested end-to-end
yet. If plots come out wrong, try rebuilding with `LIBGL_ALWAYS_SOFTWARE=1`
set in the Dockerfile's `ENV`, or adjust the `-screen` depth in
`entry-xpra.sh`'s `--xvfb=` argument.

**Password-protect it** (recommended even for local use if others share the
machine):

```bash
OPENPTV2_XPRA_PASSWORD=something ./docker/run-xpra.sh
```

Without a password, `entry-xpra.sh` still starts (bound to `127.0.0.1` only
by `run-xpra.sh`, so it's not reachable off the machine) but logs a warning.
**It refuses to start at all** if bound to a non-localhost address with
neither `OPENPTV2_XPRA_PASSWORD` nor `OPENPTV2_XPRA_ALLOW_INSECURE=1` set —
see "Security" below for why.

## Cross-platform local filesystem access

The image itself is a plain Linux container — no per-OS build variants —
and reaching host data always works the same underlying way, a Docker bind
mount (`-v host_path:/data`), which `run-xpra.sh` / `run-xpra.ps1` set up
for you. What differs per host OS is how Docker gets from that container
mount back to real files on your disk:

- **Linux**: native, no translation layer — the bind mount is a direct
  kernel-level mount, `-v /home/you/experiments:/data` just works. One
  wrinkle: the container runs as a non-root user baked in at UID 1000, so
  files it creates under `/data` show up on the host owned by UID 1000. If
  that's not your host user's UID, you can still read/write from inside
  the container fine, and read the files from the host, but modifying them
  directly as your host user may need `sudo chown -R $(id -u):$(id -g)
  /path/to/experiments` afterward.
- **macOS (Docker Desktop)**: the bind mount goes through Docker Desktop's
  VM, and it only shares directories you've explicitly allowed. If
  `/data` shows up empty despite mounting a real folder, check
  **Docker Desktop → Settings → Resources → File Sharing** and add the
  parent directory (or your whole home folder) to the allowed list.
- **Windows (Docker Desktop)**: two ways to run this, both fine:
  - From **WSL2 or Git Bash**: use `run-xpra.sh` as-is with Linux-style
    paths (`/mnt/c/Users/you/experiments` for a Windows `C:\Users\you\
    experiments` folder under WSL2, or the WSL-native path if your data
    already lives inside the WSL filesystem).
  - From **native PowerShell**: use `run-xpra.ps1 -DataDir
    C:\Users\you\experiments` — plain Windows paths, Docker Desktop
    translates them. Same File Sharing setting as macOS applies if the
    drive isn't already shared.

**Mount a parent folder, not one experiment at a time**, if you want to
switch between experiments from `pyptv_gui`'s own file picker without
restarting the container — e.g. mount your whole `~/ptv-experiments`
directory rather than `~/ptv-experiments/run3`. The container can only see
what was mounted at startup; a new bind mount needs a container restart.

## Deploying on GCP

Two reasonable options, depending on session length and how much you want
to manage.

### Option 1: Cloud Run (recommended to start)

Matches the pattern the repo already uses for `Dockerfile.cloud`. Cloud Run
supports WebSockets natively (what Xpra's HTML5 client uses) with no special
flags, but two settings matter for *this* image specifically, because each
running container holds one stateful GUI session (one Xvfb + one Xpra
process + one app), not a stateless request handler:

- `--concurrency=1` — each container instance must serve exactly one
  session. Cloud Run's session affinity for WebSockets is best-effort only
  ("requests could still potentially end up at different instances," per
  Google's docs), and there is no state-sync between Xpra processes, so
  concurrency above 1 risks two browser tabs fighting over one session or
  landing on different, empty ones.
- `--timeout` — Cloud Run's request timeout caps a WebSocket connection at
  **60 minutes maximum**. Fine for a working session with reconnects; not
  fine if you want to leave a long calibration session open unattended.
  If that matters, use Option 2 instead.

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/openptv2-xpra \
  -f docker/Dockerfile.xpra .

gcloud run deploy openptv2-xpra \
  --image gcr.io/YOUR_PROJECT/openptv2-xpra \
  --port 9876 \
  --concurrency 1 \
  --timeout 3600 \
  --memory 2Gi \
  --cpu 2 \
  --no-allow-unauthenticated \
  --region YOUR_REGION
```

**Auth — don't rely on the xpra password for this.** `--no-allow-unauthenticated`
means Cloud Run itself rejects any request without a valid Google-authenticated
identity, which is a real access barrier; a shared xpra password is not. Put
[Identity-Aware Proxy](https://cloud.google.com/iap) in front of the service
(IAP supports Cloud Run directly) so users log in with Google/your
organization's identity before ever reaching the container, and set
`OPENPTV2_XPRA_PASSWORD` as defense-in-depth behind that, not instead of it.

Data: Cloud Run's filesystem is ephemeral and gone when the instance scales
to zero. For anything beyond the baked demo, mount a
[Cloud Storage FUSE volume](https://cloud.google.com/run/docs/configuring/services/gcs-volume-mounts)
at `/data`, or accept that this deployment is for quick interactive checks,
not where real experiment data lives long-term.

### Option 2: a GCE VM (for long sessions or large data)

Better fit when a session needs to stay open for hours, or the dataset is
large enough that Cloud Run's ephemeral storage / 60-minute cap are real
constraints.

```bash
gcloud compute instances create-with-container openptv2-xpra-vm \
  --container-image gcr.io/YOUR_PROJECT/openptv2-xpra \
  --container-restart-policy always \
  --container-env OPENPTV2_XPRA_PASSWORD=something \
  --machine-type e2-standard-4 \
  --zone YOUR_ZONE

gcloud compute firewall-rules create allow-openptv2-xpra \
  --allow tcp:9876 \
  --source-ranges YOUR_IP/32 \
  --target-tags openptv2-xpra
```

Attach a persistent disk (or mount a bucket with `gcsfuse`) at `/data` for
real experiment data rather than relying on the container's own filesystem.
**Restrict `--source-ranges` to known IPs** (your lab, a VPN range) — a VM
with an open firewall rule and no xpra password is a GUI exposed to the
whole internet. If broader access is needed, put a load balancer with IAP
in front of the VM the same way as the Cloud Run option, rather than
widening the firewall rule.

## Security

Xpra has **no authentication by default**. CyVerse VICE gets away without
setting any in their own Dockerfiles because their platform puts an
authenticated gateway in front of every session; this image has no
equivalent built in, so `docker/entry-xpra.sh` refuses to start with no
auth configured unless `OPENPTV2_XPRA_ALLOW_INSECURE=1` is set explicitly.
Treat that env var as "I have confirmed this is not reachable from anywhere
untrusted," not a default to leave on.

## Open items

- End-to-end click-test (build → run → open in a real browser → run
  detection → confirm Chaco plots render) hasn't been done yet — do this
  before depending on it for real work.
- **Fixed, but only reasoned through, not re-verified by an actual build**:
  the first version of `Dockerfile.xpra` used a hand-written, outdated apt
  source for xpra.org's repo (`gpg --dearmor` + a manually typed `deb
  [signed-by=...] https://xpra.org/ bookworm main` line) and failed to
  build (`apt-get` exit 100) on a real attempt. Replaced with xpra.org's
  current documented method — the pre-armored key at `xpra.org/xpra.asc`
  and the actual per-codename `.sources` file fetched directly from
  `Xpra-org/xpra`'s repo — and pinned the base image to
  `python:3.12-slim-bookworm` explicitly so it can't drift from the
  codename that `.sources` file is for. This should build now; it hasn't
  been re-attempted on real hardware yet, so treat it as "should work,"
  not "confirmed," until the next build attempt.
- Multi-user session lifecycle (spin-up per user, idle teardown) isn't
  built. Both GCP options above are single-session; running this for a lab
  or a wider community needs an orchestration layer on top — CyVerse VICE's
  own platform (beyond just their Dockerfiles) is worth studying if that
  becomes the next step.
