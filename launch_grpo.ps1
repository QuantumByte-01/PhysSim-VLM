$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = "cmd.exe"
$pinfo.Arguments = "/c `"C:\Users\Swastik R\Documents\Personal_Projects\VLM and Physics\run_grpo.bat`""
$pinfo.UseShellExecute = $true
$pinfo.WindowStyle = "Minimized"
$p = [System.Diagnostics.Process]::Start($pinfo)
Write-Output "PID: $($p.Id)"
