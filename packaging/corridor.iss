; Inno Setup script for Corridor.
;
; Two deliberate choices:
;   * PrivilegesRequired=lowest installs per-user, so a researcher on a managed
;     lab machine does not need an administrator to try it.
;   * Nothing under {localappdata}\Corridor is ever removed. That is where
;     projects and results live, so an upgrade - and even an uninstall - must
;     leave a user's analyses intact.

#define AppName        "Corridor"
#define AppVersion     "1.0.0"
#define AppPublisher   "Corridor"
#define AppExeName     "Corridor.exe"
#define AppId          "{6F4B1D0E-2C7A-4F63-9E11-0A5C8B3D7A21}"

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
DisableReadyPage=no
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\build\installer
OutputBaseFilename=Corridor-{#AppVersion}-Setup
SetupIconFile=..\src\corridor\assets\corridor.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
AppSupportURL=https://github.com/HI1098765432/corridor
AppUpdatesURL=https://github.com/HI1098765432/corridor/releases
LicenseFile=..\LICENSE
CloseApplications=yes
CloseApplicationsFilter=*.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "associatetif"; Description: "Open TIFF stacks with {#AppName}"; GroupDescription: "File types:"; Flags: unchecked

[Files]
Source: "..\build\dist\Corridor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Only registered when the user asks for the association.
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids"; ValueType: string; \
    ValueName: "Corridor.TiffStack"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associatetif
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; \
    ValueName: "Corridor.TiffStack"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associatetif
Root: HKCU; Subkey: "Software\Classes\Corridor.TiffStack"; ValueType: string; \
    ValueName: ""; ValueData: "Time-lapse stack"; Flags: uninsdeletekey; Tasks: associatetif
Root: HKCU; Subkey: "Software\Classes\Corridor.TiffStack\DefaultIcon"; ValueType: string; \
    ValueName: ""; ValueData: "{app}\{#AppExeName},0"; Tasks: associatetif
Root: HKCU; Subkey: "Software\Classes\Corridor.TiffStack\shell\open\command"; ValueType: string; \
    ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%1"""; Tasks: associatetif

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Open {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove only what the installer created. Projects, results and settings under
; {localappdata}\Corridor are the user's data and are deliberately left alone.
Type: filesandordirs; Name: "{app}\_internal\__pycache__"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Corridor');
    if DirExists(DataDir) then
      MsgBox('Your Corridor projects and results have been kept in:' + #13#10 + #13#10 +
             DataDir + #13#10 + #13#10 +
             'Delete that folder yourself if you no longer need them.',
             mbInformation, MB_OK);
  end;
end;
