# Creates the key that signs Open Player's updates (run once, by the maintainer).
#   powershell -ExecutionPolicy Bypass -File tools\make-update-key.ps1 -Out C:\path\to\private\folder
# Writes <Out>\open-player-update-key.pem: an ECDSA P-256 private key, ENCRYPTED with a passphrase you type here
# (AES-256). Keep that file in a private place only (the private Open-Player-Keys repo) and the passphrase in your
# password manager. Losing both means installed players can't verify new updates.
# Also writes player\update_pubkey.h, the public half the player checks signatures with (safe to publish).
param([Parameter(Mandatory = $true)][string]$Out)
$ErrorActionPreference = "Stop"
$openssl = "C:\Program Files\Git\usr\bin\openssl.exe"  # ships with Git for Windows
if (-not (Test-Path $openssl)) { throw "OpenSSL not found at $openssl (install Git for Windows)" }
$key = Join-Path $Out "open-player-update-key.pem"
if (Test-Path $key) { throw "$key already exists: refusing to replace the update key" }
New-Item -ItemType Directory -Force $Out | Out-Null

function Read-Secret($prompt) {
    $s = Read-Host -AsSecureString $prompt
    $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
    try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }
}
$p1 = Read-Secret "Passphrase for the update key (12+ characters)"
$p2 = Read-Secret "Same passphrase again"
if ($p1 -ne $p2) { throw "The passphrases differ" }
if ($p1.Length -lt 12) { throw "Use at least 12 characters" }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("op-key-" + [guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
$env:OP_UPDATE_KEY_PASS = $p1  # handed to OpenSSL by name (env:), never on a command line
try {
    & $openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -aes-256-cbc -pass env:OP_UPDATE_KEY_PASS -out $key
    if ($LASTEXITCODE) { throw "key generation failed" }
    if ((Get-Content $key -TotalCount 1) -ne "-----BEGIN ENCRYPTED PRIVATE KEY-----") { Remove-Item $key; throw "key not encrypted" }

    & $openssl pkey -in $key -passin env:OP_UPDATE_KEY_PASS -pubout -out "$tmp\pub.pem"
    & $openssl pkey -pubin -in "$tmp\pub.pem" -outform DER -out "$tmp\pub.der"
    if ($LASTEXITCODE) { throw "public key export failed" }
    $der = [IO.File]::ReadAllBytes("$tmp\pub.der")
    if ($der.Length -ne 91 -or $der[26] -ne 4) { throw "unexpected public key format" }
    $xy = $der[27..90]  # X || Y after the 04 (uncompressed point) prefix

    # self-test: sign a random message with the encrypted key, verify with the public key
    $msg = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($msg)
    [IO.File]::WriteAllBytes("$tmp\msg.bin", $msg)
    & $openssl dgst -sha256 -sign $key -passin env:OP_UPDATE_KEY_PASS -out "$tmp\sig.der" "$tmp\msg.bin"
    $verify = & $openssl dgst -sha256 -verify "$tmp\pub.pem" -signature "$tmp\sig.der" "$tmp\msg.bin"
    if ($verify -ne "Verified OK") { throw "self-test failed: $verify" }

    $hex = ($xy | ForEach-Object { "0x{0:x2}" -f $_ }) -join ", "
    $h = @"
#pragma once
// Public half of Open Player's update-signing key (ECDSA P-256, X || Y). Made by tools/make-update-key.ps1;
// the encrypted private key is kept privately by the maintainer. Updates must carry a signature by that key.
static const unsigned char UPDATE_PUBKEY[64] = {$hex};
"@
    Set-Content -Path (Join-Path $PSScriptRoot "..\player\update_pubkey.h") -Value $h -Encoding ascii
    Write-Host "key OK: $key (encrypted), player\update_pubkey.h written"
    Write-Host "Now store the passphrase in your password manager."
} finally {
    Remove-Item env:OP_UPDATE_KEY_PASS -ErrorAction SilentlyContinue
    $p1 = $p2 = $null
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
