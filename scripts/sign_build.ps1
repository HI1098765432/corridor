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

$targets = @(
  (Join-Path $Root "build\dist\Corridor\Corridor.exe"),
  (Join-Path $Root "build\installer\Corridor-1.0.0-Setup.exe")
) | Where-Object { Test-Path $_ }

foreach ($target in $targets) {
  Write-Output ""
  Write-Output "signing $target"
  & $signtool.FullName sign /fd SHA256 /td SHA256 /tr $timestamp `
      /sha1 $cert.Thumbprint "$target"
  if ($LASTEXITCODE -ne 0) { throw "signing failed for $target" }
  & $signtool.FullName verify /pa /v "$target" 2>&1 | Select-String -Pattern "Successfully verified|Hash of file|Signing Certificate|The signature is timestamped" | ForEach-Object { "  $_" }
}

Write-Output ""
Write-Output "THUMBPRINT (publish this alongside the checksum):"
Write-Output "  $($cert.Thumbprint)"
