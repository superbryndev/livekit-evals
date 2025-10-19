# Publishing to PyPI

## 🤖 Automatic Publishing (Default)

The package **automatically publishes to PyPI** when you push to `main`. Version numbers are auto-incremented.

### One-Time Setup

#### 1. Create PyPI Account
- Go to: https://pypi.org/account/register/
- Complete registration

#### 2. Generate PyPI API Token
- Go to: https://pypi.org/manage/account/token/
- Click "Add API token"
- Token name: `livekit-evals-github`
- Scope: "Entire account"
- Click "Create token"
- **Copy the token** (starts with `pypi-...`)

#### 3. Add Token to GitHub Secrets
- Go to GitHub repo → **Settings** → **Secrets and variables** → **Actions**
- Click "**New repository secret**"
- Name: `PYPI_API_TOKEN`
- Value: Paste your PyPI token
- Click "**Add secret**"

### How to Publish

Just push your changes to `main` with the right commit message:

```bash
git add .

# For bug fixes (patch: 0.1.0 → 0.1.1)
git commit -m "Fix transcript filtering bug"

# For new features (minor: 0.1.0 → 0.2.0)
git commit -m "feat: Add custom webhook URL support"
# OR
git commit -m "[minor] Add new feature"

# For breaking changes (major: 0.1.0 → 1.0.0)
git commit -m "[major] Complete API redesign"
# OR
git commit -m "BREAKING CHANGE: Rename all functions"

git push origin main
```

**That's it!** The workflow will:
1. ✅ Read your commit message
2. ✅ Auto-increment version appropriately
3. ✅ Build the package
4. ✅ Publish to PyPI
5. ✅ Commit the version bump back to repo

Check the **Actions** tab in GitHub to watch it publish.

### Version Bump Keywords

The workflow detects version bump type from your commit message:

| Keywords | Bump Type | Example | Result |
|----------|-----------|---------|--------|
| `feat:` or `[minor]` | **Minor** | `feat: Add retry logic` | 0.1.0 → 0.2.0 |
| `[major]` or `BREAKING CHANGE` | **Major** | `[major] Remove old API` | 0.2.0 → 1.0.0 |
| (default) | **Patch** | `Fix bug in webhook` | 0.1.1 → 0.1.2 |

**Examples:**
```bash
# Patch (bug fix) - DEFAULT
git commit -m "Fix filtering empty turns"
# → 0.1.0 → 0.1.1

# Minor (new feature)
git commit -m "feat: Add SIP trunk detection"
# → 0.1.1 → 0.2.0

# Major (breaking change)
git commit -m "[major] Rename create_handler to create_webhook_handler"
# → 0.2.0 → 1.0.0
```

### Monitoring

- **GitHub Actions**: Repo → Actions tab → "Publish to PyPI" workflow
- **PyPI Page**: https://pypi.org/project/livekit-evals/
- **Verify Install**: `pip install --upgrade livekit-evals`

---

## 🔧 Manual Publishing (Alternative)

If you prefer manual control:

### 1. Install Tools

```bash
pip install build twine bump2version
```

### 2. Bump Version

```bash
# Patch (0.1.0 → 0.1.1) - bug fixes
bump2version patch

# Minor (0.1.0 → 0.2.0) - new features
bump2version minor

# Major (0.1.0 → 1.0.0) - breaking changes
bump2version major
```

### 3. Build and Upload

```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

### 4. Push Version Bump

```bash
git push origin main
```

---

## 🐛 Troubleshooting

### "Invalid or non-existent authentication"
- Check `PYPI_API_TOKEN` secret is set in GitHub
- Token may have expired - regenerate it
- Make sure token has correct permissions

### "File already exists"
- Version already published to PyPI
- Version number wasn't incremented
- Check that auto-bump worked (see Actions logs)

### Workflow Doesn't Trigger
- Only triggers on `main` branch pushes
- Only when these files change:
  - `livekit_evals/**`
  - `pyproject.toml`
  - `MANIFEST.in`
- Check Actions tab for errors

### Manual Bump Not Working
```bash
# Install bump2version
pip install bump2version

# Check current version
grep version pyproject.toml

# Bump version
bump2version patch
```

---

## 📋 Publishing Checklist

- [ ] Make your code changes
- [ ] Update `CHANGELOG.md` with changes
- [ ] Test locally: `pip install -e .`
- [ ] Commit: `git commit -m "Description"`
- [ ] Push to main: `git push origin main`
- [ ] Watch GitHub Actions (Actions tab)
- [ ] Verify on PyPI: https://pypi.org/project/livekit-evals/
- [ ] Test install: `pip install --upgrade livekit-evals`

---

## 🎯 Quick Reference

### Patch Release (Bug Fixes) - DEFAULT
```bash
git add .
git commit -m "Fix bug description"
git push origin main
# → Bumps: 0.1.0 → 0.1.1
```

### Minor Release (New Features)
```bash
git add .
git commit -m "feat: Add new feature"
# OR: git commit -m "[minor] Add new feature"
git push origin main
# → Bumps: 0.1.0 → 0.2.0
```

### Major Release (Breaking Changes)
```bash
git add .
git commit -m "[major] Breaking change description"
# OR: git commit -m "BREAKING CHANGE: Description"
git push origin main
# → Bumps: 0.2.0 → 1.0.0
```

### Completely Manual
```bash
# Manually control version
bump2version major  # or minor, or patch
python -m build
twine upload dist/*
git push origin main
```

---

**Simple workflow:** Your commit message controls the version! 🚀

- Regular commit → Patch (0.1.0 → 0.1.1)
- `feat:` or `[minor]` → Minor (0.1.0 → 0.2.0)
- `[major]` or `BREAKING CHANGE` → Major (0.2.0 → 1.0.0)
