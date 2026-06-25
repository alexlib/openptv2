# Documentation Workflow & GitHub Pages Deployment

This guide explains how to add, edit, preview, and deploy documentation for OpenPTV2.

---

## 1. Documentation Structure

All documentation resides in the `docs/` folder at the root of the repository:
```text
docs/
├── index.md                 # Documentation home page
├── installation.md          # Platform & VM setup instructions
├── first_steps.md           # Quick start scripting and GUI guides
├── tests.md                 # Running and managing test suites
├── developer_guide/         # Developer guides
│   ├── building.md          # Build instructions
│   ├── documentation_workflow.md # This guide
│   └── packaging_and_releases.md # Wheel & releasing guide
├── tutorials/               # User tutorials
│   ├── getting_started_tutorial.md # End-to-end tutorial
│   ├── tracking_visualization.md
│   └── tracking_debug_visualization.md
└── HOW_TO_TEST_GUI.md       # GUI testing guide
```

---

## 2. Editing and Updating Documents

### Step 1: Create or Modify a Markdown File
You can edit any existing `.md` file inside the `docs/` folder or create a new one.

### Step 2: Register in `mkdocs.yml` (If adding a new page)
If you create a new Markdown file, you **must** register it in the `nav` section of `mkdocs.yml` at the repository root so it appears in the website sidebar:

```yaml
nav:
  - Home: index.md
  - Installation & Setup: installation.md
  - Developer Guide:
    - Building from Source: developer_guide/building.md
    - Documentation Workflow: developer_guide/documentation_workflow.md # Added here
```

---

## 3. Previewing Your Changes Locally

Before pushing changes to GitHub, it is best to build and run the documentation server locally to verify spelling, layout, and link paths:

```bash
# 1. Install development documentation utilities
uv sync --extra dev

# 2. Run the local development server
uv run mkdocs serve
```

This starts a server at **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**. The local server dynamically watches your `docs/` directory and will automatically refresh and re-render the pages in your browser as you make modifications!

---

## 4. Triggering Automated Deployment to GitHub Pages

Documentation hosting is fully automated. To render and publish updates to the live site at **[https://alexlib.github.io/openptv2/](https://alexlib.github.io/openptv2/)**:

### Option A: Automatic Deployment on Push (Recommended)
Our CI/CD workflow (`.github/workflows/deploy_docs.yml`) is configured to listen for updates:
1. When you push your modifications to the **`main`** or **`master`** branches:
   ```bash
   git add .
   git commit -m "docs: describe documentation workflow and update pages"
   git push origin main
   ```
2. The GitHub Actions runner will automatically spin up, install the dependencies, build the static site, and force-push the compiled output directly to the repository's `gh-pages` branch.

### Option B: Manual Workflow Dispatch
If you need to manually force a re-render and deployment without pushing a new commit:
1. Navigate to your repository on GitHub: `https://github.com/alexlib/openptv2`
2. Click on the **Actions** tab at the top.
3. Select **Deploy Docs** from the list of workflows on the left.
4. Click the **Run workflow** dropdown on the right, select your branch (e.g., `main`), and click the green **Run workflow** button.

---

## 5. Verifying the Rendered Website

Once the GitHub Actions workflow successfully finishes, the generated documentation site will render and serve live at:

👉 **[https://alexlib.github.io/openptv2/](https://alexlib.github.io/openptv2/)**

> [!TIP]
> If you make changes and do not see them on the live website immediately, clear your browser cache (using `Ctrl+F5` or `Cmd+Shift+R`) or check the GitHub Actions logs to make sure the **Deploy Docs** workflow run was successfully completed.
