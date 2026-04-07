; Inno Setup script for packaging the MicroSeg desktop build into a single installer.
; Run after PyInstaller produces dist/MicroSegDesktop.

#define AppName "MicroSeg Desktop"
#define AppVersion "0.22.0"
#define Publisher "MicroSeg"
#define RepoRoot "..\..\.."
#define DistRoot "..\..\..\dist\MicroSegDesktop"
#define OutputRoot "..\..\..\dist\installer"

[Setup]
AppId={{CD67B2DF-8D42-4E95-840A-7053702299A0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile={#RepoRoot}\LICENSE
OutputDir={#OutputRoot}
OutputBaseFilename=MicroSegDesktop_{#AppVersion}_offline_setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "{#DistRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\MicroSegDesktop.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\MicroSegDesktop.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MicroSegDesktop.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

