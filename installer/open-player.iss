; Open Player's Windows installer (Inno Setup 6). Built by installer\build-installer.bat, which passes
; /DBuild=<build number> and /DVCRedist=<folder with the VC++ runtime DLLs>.
; Per user (no admin), into %LOCALAPPDATA%\Programs\Open Player. Never contains TensorRT or a Pixel Flow model.

#ifndef Build
  #error Build the installer with installer\build-installer.bat
#endif

[Setup]
AppId={{7C9E4A1B-3F2D-4E8A-9B6C-5D1E2F3A4B5C}
AppName=Open Player
AppVersion=build {#Build}
AppVerName=Open Player build {#Build}
AppPublisher=Open Player
AppPublisherURL=https://github.com/SpaceJamp/Open-Player
AppSupportURL=https://github.com/SpaceJamp/Open-Player/issues
AppUpdatesURL=https://github.com/SpaceJamp/Open-Player/releases
VersionInfoVersion=1.0.0.{#Build}
DefaultDirName={localappdata}\Programs\Open Player
DisableDirPage=auto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=out
OutputBaseFilename=OpenPlayer-Setup-{#Build}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\open-player.exe
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; an update runs this while the player may still be closing
CloseApplications=yes
RestartApplications=no
; tells Explorer about the Open with entries at once
ChangesAssociations=yes

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
Name: openwith; Description: "Add Open Player to ""Open with"" for video files"

[Files]
Source: "..\build-win\open-player.exe"; DestDir: "{app}"; Flags: ignoreversion
; FFmpeg, SDL, dav1d and NVIDIA's RTX VSR runtime; TensorRT is an optional add-on, downloaded from NVIDIA
Source: "..\build-win\*.dll"; Excludes: "nvinfer*,nvonnxparser*"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#VCRedist}\msvcp140*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#VCRedist}\vcruntime140*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\*"; Excludes: "*.svg"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs
Source: "..\LICENSE"; DestDir: "{app}"
Source: "..\LICENSE-EXCEPTION"; DestDir: "{app}"
Source: "THIRD-PARTY-NOTICES.txt"; DestDir: "{app}"
; the training scripts (Settings > Training runs them on the Training add-on's Python, downloaded on first use)
Source: "..\train.py"; DestDir: "{app}\training"; Flags: ignoreversion
Source: "..\model.py"; DestDir: "{app}\training"; Flags: ignoreversion
Source: "..\train_diag.py"; DestDir: "{app}\training"; Flags: ignoreversion
Source: "..\export.py"; DestDir: "{app}\training"; Flags: ignoreversion
; tells the player it is an installed copy: updates on, the model looked for in %LOCALAPPDATA%\Open Player
Source: "installed.txt"; DestDir: "{app}"

[UninstallDelete]
; Python's bytecode cache, should a training script ever have written one
Type: filesandordirs; Name: "{app}\training\__pycache__"

[Icons]
Name: "{autoprograms}\Open Player"; Filename: "{app}\open-player.exe"
Name: "{autodesktop}\Open Player"; Filename: "{app}\open-player.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "Open Player"; Flags: uninsdeletekey; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\shell\open\command"; ValueType: string; ValueData: """{app}\open-player.exe"" ""%1"""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".mp4"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".mkv"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".webm"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".mov"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".avi"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".m4v"; ValueData: ""; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\Applications\open-player.exe\SupportedTypes"; ValueType: string; ValueName: ".ts"; ValueData: ""; Tasks: openwith
; an empty string value (not ValueType: none, which writes no value at all) puts it in each type's Open with menu
Root: HKA; Subkey: "Software\Classes\.mp4\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.mkv\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.webm\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.mov\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.avi\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.m4v\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKA; Subkey: "Software\Classes\.ts\OpenWithProgids"; ValueType: string; ValueName: "Applications\open-player.exe"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith

[Run]
Filename: "{app}\open-player.exe"; Description: "Start Open Player"; Flags: nowait postinstall skipifsilent
; after an update the player comes back by itself (the updater passes /RELAUNCH=1; other silent installs don't)
Filename: "{app}\open-player.exe"; Flags: nowait; Check: RelaunchAsked

[Code]
function RelaunchAsked: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Data: String;
begin
  Data := ExpandConstant('{localappdata}\Open Player');
  if (CurUninstallStep = usPostUninstall) and DirExists(Data) and not UninstallSilent then
    if MsgBox('Also delete your Pixel Flow model, your training and the TensorRT and Training add-ons in' + #13#10 + Data + '?',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(Data, True, True, True);
end;
