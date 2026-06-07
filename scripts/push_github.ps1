# Push Pearls AQI to GitHub (run after: gh auth login)
param(
    [string]$RepoName = "pearls-aqi-predictor",
    [string]$Visibility = "public"
)

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .git)) {
    Write-Error "Not a git repo. Run from project root."
    exit 1
}

$status = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "GitHub CLI not logged in. Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Write-Host "Creating GitHub repo and pushing..."
    gh repo create $RepoName --$Visibility --source=. --remote=origin --push
} else {
    Write-Host "Remote exists: $remote"
    git push -u origin main
}

Write-Host "Done. View workflows at: https://github.com/$(gh api user -q .login)/$RepoName/actions"
