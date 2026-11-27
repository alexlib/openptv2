# Writing a course page or a recipe

The course (`docs/course/`) teaches the basics step by step on data that ships with the repository; recipes (`docs/recipes/`)
start from a situation and link to the course. The plan and the reasons are in `docs/plans/2026-11-27-documentation-rewrite.md`.

## The shape of a course page

```text
# N. Title: a plain statement of what the reader gets
**Time:** ...  **You need:** ...
## What this step does          3 sentences, no jargon
## (the idea, with a picture)   only if the step needs one
## Do it                        tabs: "Command line" first, "Graphical interface" second
## What you should see          the real output, in a ```text block
## Check that it worked         a number or a picture with a pass/fail rule
## If it did not work           a table: what you see | cause | what to do
[Next: ... →]
```

## Rules

1. One sentence per idea. Explain a term the first time (link the [glossary](../course/glossary.md)); code names go in parentheses.
2. Every command is copy-paste and has the output it prints. **The output block is tested**: a `python -c "..."` block followed by
   a `text` block under "What you should see" is run by `tests/unit/test_course_smoke.py` and compared line by line. Regenerate
   the block from a real run when the code changes; never type it by hand.
3. Every number has a rule ("good: below 0.3 px").
4. Use the bundled `test_data/test_cavity_small`. A command for the graphical interface must name a menu entry that exists in
   `src/openptv2/gui/pyptv_gui.py`.
5. No dataset, lab or person names in titles, file names or text, except as "example: ...". Name recipes by the situation.
6. No research findings in a how-to: link the reference page in one line.
7. Figures come from a script (`docs/course/make_figures.py`), not from a screenshot that goes stale.

Check before you commit: `uv run mkdocs build --strict` and `uv run pytest tests/unit/test_course_smoke.py`.
