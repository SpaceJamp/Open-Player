# Builds, signs and publishes an Open Player release (run by the maintainer, from a clean main).
#   powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Notes "What changed"
# 1. refuses a modified tree, a branch other than main, or a build without a number
# 2. builds the installer (installer\build-installer.bat)
# 3. writes update.json {build, file, sha256, notes} and signs it with the update key (asks for the passphrase)
# 4. checks the signature against the public key built into the player (player\update_pubkey.h)
# 5. publishes the GitHub release v<build> with the three files, only after you type "yes"
# -DryRun skips the git checks and publishing (for testing, e.g. with a throwaway -Key and -PublicKeyPem).
param(
    [string]$Notes = "",
    [string]$Key = "C:\Users\Yeti\Documents\Open-Player-Keys\open-player-update-key.pem",
    [string]$PublicKeyPem = "",  # default: the player's own key, from player\update_pubkey.h
    [switch]$DryRun,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$openssl = "C:\Program Files\Git\usr\bin\openssl.exe"
if (-not (Test-Path $Key)) { throw "update key not found: $Key" }

if (-not $DryRun) {
    if (git status --porcelain --untracked-files=no) { throw "uncommitted changes: commit first" }
    if ((git rev-parse --abbrev-ref HEAD) -ne "main") { throw "releases are made from main only" }
}
if (-not $SkipBuild) {
    cmd /c "installer\build-installer.bat"
    if ($LASTEXITCODE) { throw "installer build failed" }
}
$ver = & "build-win\open-player.exe" --version
if ($ver -notmatch '^Open Player build (\d+) \(([0-9a-f]+)(-modified)?\)$') { throw "no build number in: $ver" }
$build = [int]$Matches[1]
if ($Matches[3] -and -not $DryRun) { throw "this build has uncommitted changes: $ver" }
if ($ver -match "TEST") { throw "refusing a TEST build" }
$file = "OpenPlayer-Setup-$build.exe"
$installer = Join-Path $root "installer\out\$file"
if (-not (Test-Path $installer)) { throw "missing $installer" }

$out = Join-Path $root "installer\out\release-$build"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item $installer $out -Force
$sha = (Get-FileHash $installer -Algorithm SHA256).Hash.ToLower()
$notesJson = $Notes.Replace('\', '\\').Replace('"', '\"').Replace("`r", "").Replace("`n", '\n').Replace("`t", '\t')
if ($notesJson -match '[\x00-\x1f]') { throw "notes may not contain control characters" }
$json = '{"build": ' + $build + ', "file": "' + $file + '", "sha256": "' + $sha + '", "notes": "' + $notesJson + '"}'
$jsonPath = Join-Path $out "update.json"
[IO.File]::WriteAllBytes($jsonPath, [Text.Encoding]::UTF8.GetBytes($json))  # exact bytes: no BOM, no newline

# sign (OpenSSL asks for the key's passphrase), then DER (r, s) -> raw r || s, base64
$der = Join-Path $out "update.json.der"
& $openssl dgst -sha256 -sign $Key -out $der $jsonPath
if ($LASTEXITCODE) { throw "signing failed" }
$d = [IO.File]::ReadAllBytes($der)
function Trim-Zeros([byte[]]$x) {  # leading zero bytes off (Windows PowerShell has no Select-Object -SkipWhile)
    $k = 0
    while ($k -lt $x.Count - 1 -and $x[$k] -eq 0) { $k++ }
    return [byte[]]$x[$k..($x.Count - 1)]
}
function Read-Int([byte[]]$b, [ref]$i) {
    if ($b[$i.Value] -ne 2) { throw "unexpected signature format" }
    $n = $b[$i.Value + 1]; $v = $b[($i.Value + 2)..($i.Value + 1 + $n)]; $i.Value += 2 + $n
    $v = @(Trim-Zeros $v)  # drop DER's sign padding
    if ($v.Count -gt 32) { throw "unexpected signature format" }
    return [byte[]](@(,0 * (32 - $v.Count)) + $v)
}
if ($d[0] -ne 0x30) { throw "unexpected signature format" }
$i = 2
$rs = (Read-Int $d ([ref]$i)) + (Read-Int $d ([ref]$i))
[IO.File]::WriteAllText((Join-Path $out "update.json.sig"), [Convert]::ToBase64String([byte[]]$rs))
Remove-Item $der

# check against the key the player trusts (not just the key we signed with)
if (-not $PublicKeyPem) {
    $hex = [regex]::Matches((Get-Content "player\update_pubkey.h" -Raw), '0x([0-9a-f]{2})') | ForEach-Object { $_.Groups[1].Value }
    if ($hex.Count -ne 64) { throw "player\update_pubkey.h doesn't hold a 64-byte key" }
    $spki = "3059301306072a8648ce3d020106082a8648ce3d03010703420004" + ($hex -join "")
    $bytes = [byte[]]($spki -split '(..)' | Where-Object { $_ } | ForEach-Object { [Convert]::ToByte($_, 16) })
    $PublicKeyPem = Join-Path $out "player-pubkey.pem"
    Set-Content $PublicKeyPem ("-----BEGIN PUBLIC KEY-----`n" + [Convert]::ToBase64String($bytes) + "`n-----END PUBLIC KEY-----`n") -Encoding ascii
}
$check = Join-Path $out "check.der"
$sigBytes = [Convert]::FromBase64String((Get-Content (Join-Path $out "update.json.sig") -Raw))
function To-Der([byte[]]$x) { $x = @(Trim-Zeros $x); if ($x[0] -ge 0x80) { $x = @([byte]0) + $x }; [byte[]](@(2, $x.Count) + $x) }
$r = To-Der $sigBytes[0..31]; $s = To-Der $sigBytes[32..63]
[IO.File]::WriteAllBytes($check, [byte[]](@(0x30, ($r.Count + $s.Count)) + $r + $s))
$v = & $openssl dgst -sha256 -verify $PublicKeyPem -signature $check $jsonPath
Remove-Item $check
if ($v -ne "Verified OK") { throw "the signature doesn't verify with the player's key: $v" }
Write-Host "signed and verified: $out (update.json, update.json.sig, $file)"

if ($DryRun) { Write-Host "dry run: not published"; return }
$answer = Read-Host "Publish GitHub release v$build with these 3 files? Type yes"
if ($answer -ne "yes") { Write-Host "not published"; return }
gh release create "v$build" "$out\$file" "$out\update.json" "$out\update.json.sig" --title "Open Player build $build" --notes $(if ($Notes) { $Notes } else { "Open Player build $build" })
