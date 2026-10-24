# Installs the latest Temper on Windows:
#   irm https://temper.devops-monk.com/install.ps1 | iex
#
# Downloads the installer for your processor (x64 or ARM64), checks it against the SHA-256 that
# GitHub publishes for the release, and installs Temper for the current user (no administrator
# rights needed).
#
# Settings (environment variables):
#   TEMPER_VERSION   a release tag such as v0.1.0 (default: the latest release)

& {
  $ErrorActionPreference = "Stop"
  # Invoke-WebRequest's progress bar slows downloads down a lot in Windows PowerShell.
  $ProgressPreference = "SilentlyContinue"
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

  $repo = "devops-monk/temper"
  $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
  $suffix = switch ($arch) {
    "ARM64" { "_arm64-setup.exe" }
    "AMD64" { "_x64-setup.exe" }
    default { throw "Temper has no build for $arch processors." }
  }

  $api = if ($env:TEMPER_VERSION) { "https://api.github.com/repos/$repo/releases/tags/$($env:TEMPER_VERSION)" } else { "https://api.github.com/repos/$repo/releases/latest" }
  Write-Host "Finding the latest Temper release..."
  try {
    $headers = @{ Accept = "application/vnd.github+json" }
    # GITHUB_TOKEN is optional; it only lifts GitHub's rate limit for unauthenticated requests.
    if ($env:GITHUB_TOKEN) { $headers.Authorization = "Bearer $($env:GITHUB_TOKEN)" }
    $release = Invoke-RestMethod -Uri $api -Headers $headers -UseBasicParsing
  } catch {
    throw "Temper install failed: couldn't reach GitHub. Check your internet connection and try again."
  }

  $asset = $release.assets | Where-Object { $_.name.EndsWith($suffix) } | Select-Object -First 1
  if (-not $asset) { throw "Temper install failed: the release has no installer for Windows $arch yet." }

  $file = Join-Path ([IO.Path]::GetTempPath()) $asset.name
  Write-Host "Downloading Temper $($release.tag_name) for Windows ($arch)..."
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $file -UseBasicParsing

  try {
    if ($asset.digest -and $asset.digest.StartsWith("sha256:")) {
      $actual = (Get-FileHash -Algorithm SHA256 -Path $file).Hash.ToLower()
      if ($actual -ne $asset.digest.Substring(7)) {
        throw "Temper install failed: the download doesn't match its published checksum. Nothing was installed."
      }
    } else {
      Write-Host "Note: GitHub published no checksum for this file, so it wasn't verified."
    }

    Write-Host "Installing..."
    # /S runs the installer without its wizard; it installs for the current user only.
    $proc = Start-Process -FilePath $file -ArgumentList "/S" -Wait -PassThru
    if ($proc.ExitCode -ne 0) { throw "Temper install failed: the installer stopped with code $($proc.ExitCode)." }
  } finally {
    Remove-Item -Force $file -ErrorAction SilentlyContinue
  }

  $exe = Join-Path $env:LOCALAPPDATA "Temper\temper.exe"
  Write-Host ""
  Write-Host "Temper $($release.tag_name) is installed. Open it from the Start menu."
  if (Test-Path $exe) { Write-Host "It is at $exe" }
}
