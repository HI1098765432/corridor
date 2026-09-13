# Launch the installed application and capture its real window.
#
# PrintWindow with PW_RENDERFULLCONTENT is used rather than a screen grab so
# the capture is the window's own pixels: unaffected by display scaling, by
# what else is on screen, or by whether the window is occluded.

param(
  [string]$Exe = "$env:LOCALAPPDATA\Programs\Corridor\Corridor.exe",
  [string]$Out = "C:\Users\ironb\Projects\ConfinedMig\data\_ui\installed.png",
  [int]$WaitSeconds = 45,
  [string]$OpenFile = ""
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win {
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
"@

$args = @()
if ($OpenFile) { $args += $OpenFile }
$proc = if ($args.Count) { Start-Process -FilePath $Exe -ArgumentList $args -PassThru }
        else { Start-Process -FilePath $Exe -PassThru }
Write-Output "launched pid $($proc.Id), waiting for a window..."

$handle = [IntPtr]::Zero
for ($i = 0; $i -lt $WaitSeconds * 2; $i++) {
  Start-Sleep -Milliseconds 500
  $proc.Refresh()
  if ($proc.HasExited) { Write-Output "process exited early, code $($proc.ExitCode)"; exit 1 }
  if ($proc.MainWindowHandle -ne [IntPtr]::Zero -and [Win]::IsWindowVisible($proc.MainWindowHandle)) {
    $handle = $proc.MainWindowHandle
    break
  }
}
if ($handle -eq [IntPtr]::Zero) { Write-Output "no window appeared within $WaitSeconds s"; exit 2 }

Write-Output "window: '$($proc.MainWindowTitle)'"
# Give Qt a moment to finish its first paint.
Start-Sleep -Seconds 3
[Win]::SetForegroundWindow($handle) | Out-Null
Start-Sleep -Milliseconds 800

$rect = New-Object Win+RECT
[Win]::GetWindowRect($handle, [ref]$rect) | Out-Null
$w = $rect.R - $rect.L
$h = $rect.B - $rect.T
Write-Output "size: ${w}x${h}"

$bmp = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
# 2 = PW_RENDERFULLCONTENT, required for hardware-composited (Qt) windows.
[Win]::PrintWindow($handle, $hdc, 2) | Out-Null
$g.ReleaseHdc($hdc)
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output "saved $Out"

Start-Sleep -Seconds 1
if (-not $proc.HasExited) { $proc.CloseMainWindow() | Out-Null; Start-Sleep -Seconds 3 }
if (-not $proc.HasExited) { $proc.Kill() }
Write-Output "closed"
