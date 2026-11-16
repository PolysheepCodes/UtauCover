; UtauCover installer (Inno Setup 6). Built by packaging/build.py, which passes:
;   /DAppVersion=0.7.0 /DSourceDir=<the assembled UtauCover folder> /DOutputDir=dist /DOutputName=...
;
; Installs for the current user by default: %LOCALAPPDATA%\Programs\UtauCover, no administrator rights
; (like VS Code's user installer). The first page offers "Install for all users" instead, which installs
; to Program Files and asks for administrator rights; Windows doesn't allow writing to Program Files
; without them.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0D3C1E-8F7A-4C55-9E2B-1D7A2C9F4E61}
AppName=UtauCover
AppVersion={#AppVersion}
AppVerName=UtauCover {#AppVersion}
AppPublisher=UtauCover
DefaultDirName={autopf}\UtauCover
DefaultGroupName=UtauCover
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsePreviousPrivileges=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename={#OutputName}
SetupIconFile={#SourceDir}\UtauCover.ico
UninstallDisplayIcon={app}\UtauCover.exe
UninstallDisplayName=UtauCover
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ChangesEnvironment=no
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "portable.txt"

[InstallDelete]
; a previous version's Python packages: replaced wholesale so nothing stale is left behind
Type: filesandordirs; Name: "{app}\runtime"

[Icons]
Name: "{autoprograms}\UtauCover"; Filename: "{app}\UtauCover.exe"; AppUserModelID: "UtauCover.UtauCover"
Name: "{autodesktop}\UtauCover"; Filename: "{app}\UtauCover.exe"; Tasks: desktopicon; AppUserModelID: "UtauCover.UtauCover"

[Run]
Filename: "{app}\UtauCover.exe"; Description: "{cm:LaunchProgram,UtauCover}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"

[Code]
// The window needs Microsoft's WebView2 runtime. It ships with Windows 11 and current Windows 10;
// on an older Windows 10 point the user to Microsoft's (free, no-admin) download.
function WebView2Installed(): Boolean;
var v: String;
begin
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v) or
    RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', v);
  if Result then Result := (v <> '') and (v <> '0.0.0.0');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var code: Integer;
begin
  if (CurStep = ssPostInstall) and (not WebView2Installed()) and (not WizardSilent()) then
    if MsgBox('UtauCover''s window needs Microsoft Edge WebView2, which isn''t installed on this PC.' + #13#10 +
              'Open Microsoft''s download page now? (Pick "Evergreen Bootstrapper".)', mbConfirmation, MB_YESNO) = IDYES then
      ShellExec('open', 'https://developer.microsoft.com/microsoft-edge/webview2/', '', '', SW_SHOWNORMAL, ewNoWait, code);
end;
