# Sign the application and the installer.
#
# What this does and does not buy:
#
#   It DOES make every binary tamper-evident. Any modification after signing
#   breaks the signature, and the published thumbprint lets anyone check that
#   the file they downloaded is the one that was built.
#
#   It does NOT remove Windows SmartScreen's warning on first run. That needs a
#   certificate issued by an authority Windows already trusts, which has to be
#   bought and tied to a verified legal identity. Nothing in this script can
#   substitute for that, and pretending otherwise would be worse than the
#   warning.
#
# The certificate is created once and reused, so the publisher identity stays
# the same across releases.

param(
  [string]$Subject = "CN=Corridor, O=Corridor, C=GB",
  [string]$FriendlyName = "Corridor code signing",
  [string]$Root = "C:\Users\ironb\Projects\ConfinedMig"
)

$ErrorActionPreference = "Stop"

$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Filter "signtool.exe" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*x64*" } | Select-Object -First 1
if (-not $signtool) { throw "signtool.exe not found; install the Windows SDK." }
Write-Output "signtool: $($signtool.FullName)"

# Reuse an existing certificate so releases share one publisher identity.
$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
        Where-Object { $_.Subject -eq $Subject -and $_.NotAfter -gt (Get-Date) } |
        Select-Object -First 1

if (-not $cert) {
  Write-Output "creating a code signing certificate..."
  $cert = New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject $Subject `
    -FriendlyName $FriendlyName `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -KeyExportPolicy Exportable `
    -KeyUsage DigitalSignature `
    -KeyAlgorithm RSA `
    -KeyLength 3072 `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddYears(5)
}
Write-Output "certificate: $($cert.Subject)"
Write-Output "thumbprint:  $($cert.Thumbprint)"
Write-Output "valid until: $($cert.NotAfter.ToString('yyyy-MM-dd'))"

# Export the public certificate so the signature can be checked by anyone.
$certDir = Join-Path $Root "build"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null
$cerPath = Join-Path $certDir "corridor-codesign.cer"
Export-Certificate -Cert $cert -FilePath $cerPath -Force | Out-Null
Write-Output "public cert: $cerPath"

# A timestamp keeps signatures valid after the certificate expires.
$timestamp = "http://timestamp.digicert.com"

# The installer to sign is named by the build manifest, never by globbing the
# output folder. `Get-ChildItem ... | Select-Object -First 1` returns whichever
# file sorts first, so as soon as one previous release is left lying there it
# signs the OLD installer and reports success -- which is exactly what happened
# once. A release step that can silently act on last release's artefact is
# worse than one that fails.
$manifestPath = Join-Path (Join-Path $Root "build") "build_manifest.json"
if (-not (Test-Path $manifestPath)) {
  throw "No build manifest at $manifestPath. Run scripts/build_release.py first."
}
$manifest = Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$installerName = $manifest.installer.filename
if (-not $installerName) { throw "The build manifest does not name an installer." }
$installerPath = Join-Path $Root "build\installer\$installerName"
Write-Output "manifest names: $installerName (version $($manifest.version))"

$targets = @(
  (Join-Path $Root "build\dist\Corridor\Corridor.exe"),
  $installerPath
) | Where-Object { Test-Path $_ }

if ($targets.Count -lt 2) {
  Write-Output "NOTE: $installerPath is not present yet; signing the application only."
}

foreach ($target in $targets) {
  Write-Output ""
  Write-Output "signing $target"
  & $signtool.FullName sign /fd SHA256 /td SHA256 /tr $timestamp `
      /sha1 $cert.Thumbprint "$target"
  if ($LASTEXITCODE -ne 0) { throw "signing failed for $target" }
  # `signtool verify /pa` checks the chain against Windows' trusted roots, so
  # it always fails for a self-signed certificate. That is the expected result
  # and not a signing failure. What matters here is that the signature is
  # present, intact, and attributable, which Get-AuthenticodeSignature reports
  # without requiring the root to be trusted.
  $sig = Get-AuthenticodeSignature -FilePath $target
  $signer = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { "none" }
  $stamped = if ($sig.TimeStamperCertificate) { "yes" } else { "no" }
  Write-Output "  signature status : $($sig.Status)"
  Write-Output "  signer           : $signer"
  Write-Output "  timestamped      : $stamped"
  if (-not $sig.SignerCertificate) { throw "no signature was attached to $target" }
  if ($sig.Status -eq "HashMismatch") { throw "the signature does not match $target" }
  if ($sig.Status -notin @("Valid", "UnknownError")) {
    # UnknownError is what Windows reports for an untrusted (self-signed) root.
    throw "unexpected signature status for $target : $($sig.Status)"
  }
}

Write-Output ""
# Signing rewrote the installer, so the checksum in the build manifest now
# describes a file that no longer exists. Record the real one, or the publish
# step will refuse to run -- correctly.
$stamp = Join-Path (Join-Path $Root "scripts") "stamp_signature.py"
$python = Join-Path (Join-Path (Join-Path $Root ".venv") "Scripts") "python.exe"
if ((Test-Path $stamp) -and (Test-Path $python)) {
  & $python $stamp --thumbprint $cert.Thumbprint --subject $Subject --self-signed
  if ($LASTEXITCODE -ne 0) { throw "could not stamp the signature into the manifest" }
}

Write-Output "THUMBPRINT (publish this alongside the checksum):"
Write-Output "  $($cert.Thumbprint)"
