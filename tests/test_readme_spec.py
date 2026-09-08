"""Automated validation for README.md consistency with Standard Readme (spec.md).

Inspired by RichardLitt/standard-readme-preset (remark-preset-lint-standard-readme),
this test suite enforces the specification natively in Python for contributors.
"""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def get_pyproject_meta() -> dict[str, str]:
    """Parse basic project metadata from pyproject.toml."""
    try:
        import tomllib

        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
            project = data.get("project", {})
            return {
                "name": project.get("name", ""),
                "description": project.get("description", ""),
            }
    except ImportError:
        try:
            import tomli

            with open(PYPROJECT_PATH, "rb") as f:
                data = tomli.load(f)
                project = data.get("project", {})
                return {
                    "name": project.get("name", ""),
                    "description": project.get("description", ""),
                }
        except ImportError:
            # Fallback regex extraction
            text = PYPROJECT_PATH.read_text(encoding="utf-8")
            name_m = re.search(r'name\s*=\s*"([^"]+)"', text)
            desc_m = re.search(r'description\s*=\s*"([^"]+)"', text)
            return {
                "name": name_m.group(1) if name_m else "",
                "description": desc_m.group(1) if desc_m else "",
            }


def slugify(title: str) -> str:
    """GitHub markdown heading anchor slug."""
    clean = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"[-\s]+", "-", clean)


def test_readme_exists():
    assert README_PATH.is_file(), "README.md does not exist."
    assert README_PATH.stat().st_size > 0, "README.md is empty."


def test_title_and_banner():
    lines = [
        line.strip()
        for line in README_PATH.read_text(encoding="utf-8").splitlines()
    ]
    non_empty = [line for line in lines if line]

    meta = get_pyproject_meta()
    pkg_name = meta["name"]

    # Title must be first heading and mention package name
    assert non_empty[0].startswith(
        "# "
    ), f"First line must be level-1 title, got: {non_empty[0]}"
    assert (
        f"_({pkg_name})_" in non_empty[0] or pkg_name in non_empty[0]
    ), f"Title must mention package manager name '{pkg_name}'"

    # Banner must immediately follow title
    banner_match = re.match(r"!\[(.*?)\]\((.*?)\)", non_empty[1])
    assert (
        banner_match is not None
    ), f"Second element must be a banner image, got: {non_empty[1]}"
    banner_target = banner_match.group(2)
    assert (
        REPO_ROOT / banner_target
    ).exists(), f"Banner image file does not exist: {banner_target}"


def test_badges_newline_delimited():
    content = README_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]

    # Collect consecutive badge lines
    badge_lines = [
        line for line in lines if re.match(r"^\[!\[.*?\]\(.*?\)\]\(.*?\)$", line)
    ]
    assert len(badge_lines) >= 3, "Expected at least 3 badges."
    assert any(
        "standard-readme" in line for line in badge_lines
    ), "Standard Readme badge missing."


def test_short_description_matches_pyproject():
    meta = get_pyproject_meta()
    expected_desc = meta["description"]
    assert expected_desc, "Description in pyproject.toml is empty."
    assert (
        len(expected_desc) < 120
    ), f"Description exceeds 120 characters: {len(expected_desc)}"

    content = README_PATH.read_text(encoding="utf-8")
    assert (
        expected_desc in content
    ), "Short description in README.md must match pyproject.toml verbatim."

    # Must not start with >
    for line in content.splitlines():
        if expected_desc in line:
            assert not line.strip().startswith(
                ">"
            ), "Short description must not start with '> '"


def test_package_name_note_present():
    content = README_PATH.read_text(encoding="utf-8")
    meta = get_pyproject_meta()
    pkg_name = meta["name"]
    folder_name = REPO_ROOT.name

    if pkg_name != folder_name:
        assert (
            "Note on Naming" in content or "naming" in content.lower()
        ), "Discrepancy between folder name and package name must be noted in long description."


def test_section_order_and_required_sections():
    content = README_PATH.read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)

    required_order = [
        "Table of Contents",
        "Background",
        "Install",
        "Usage",
        "API",
        "Maintainers",
        "Thanks",
        "Contributing",
        "License",
    ]

    # Check required headings exist
    for req in required_order:
        assert (
            req in headings
        ), f"Required section '## {req}' is missing from README.md"

    # Verify relative order of required sections
    positions = [headings.index(req) for req in required_order]
    assert positions == sorted(
        positions
    ), f"Sections are not in required order: {[headings[p] for p in positions]}"

    # License must be strictly the last level-2 section
    assert (
        headings[-1] == "License"
    ), f"'License' must be the last section, but found: {headings[-1]}"


def test_table_of_contents_complete():
    content = README_PATH.read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)

    # ToC links
    toc_match = re.search(
        r"## Table of Contents\s*\n\n(.*?)(?=\n## |\Z)", content, re.DOTALL
    )
    assert toc_match is not None, "Table of Contents block not found."
    toc_text = toc_match.group(1)

    # Every level-2 heading after Table of Contents must appear in ToC
    subsequent_headings = headings[headings.index("Table of Contents") + 1 :]
    for h in subsequent_headings:
        slug = slugify(h)
        assert f"(#{slug})" in toc_text, f"ToC is missing entry for '## {h}' (#{slug})"


def test_no_broken_internal_anchors():
    content = README_PATH.read_text(encoding="utf-8")
    all_headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
    slugs = {slugify(h) for h in all_headings}

    anchor_links = re.findall(r"\[([^\]]+)\]\((#[^\)]+)\)", content)
    assert anchor_links, "No internal anchor links found."

    for text, anchor in anchor_links:
        target = anchor[1:]
        assert (
            target in slugs
        ), f"Broken anchor link: [{text}]({anchor}) (target '#{target}' does not match any heading)"


def test_no_broken_local_file_links():
    content = README_PATH.read_text(encoding="utf-8")
    file_links = re.findall(r"\[([^\]]+)\]\(([^#\):]+(?:#[^\)]*)?)\)", content)

    for text, path in file_links:
        # Ignore external URLs or mailto links
        if (
            path.startswith("http://")
            or path.startswith("https://")
            or path.startswith("mailto:")
        ):
            continue

        clean_path = path.split("#")[0].strip()
        if not clean_path:
            continue

        resolved = REPO_ROOT / clean_path
        assert (
            resolved.exists()
        ), f"Broken file link: [{text}]({path}) -> {resolved} not found."

if __name__ == "__main__":
    import sys

    failed = 0
    test_funcs = [
        (name, func)
        for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func)
    ]
    for name, func in test_funcs:
        try:
            func()
            print(f"[OK] {name}")
        except AssertionError as err:
            print(f"[FAIL] {name}: {err}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)
    print(f"\nAll {len(test_funcs)} README specification checks passed.")
