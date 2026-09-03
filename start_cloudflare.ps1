if (Test-Path cf.log) { Remove-Item cf.log }

Write-Host "Starting Cloudflare Tunnel... please wait." -ForegroundColor Cyan

# Start cloudflared and redirect its stderr to a file
$cf = Start-Process -FilePath "cloudflared" -ArgumentList "tunnel --url http://localhost:8080" -RedirectStandardError "cf.log" -NoNewWindow -PassThru

try {
    # Tail the log file manually
    Get-Content -Path cf.log -Wait | ForEach-Object {
        if ($_ -match "(https://[a-zA-Z0-9-]+\.trycloudflare\.com)") {
            Write-Host "
==================================================================" -ForegroundColor Magenta
            Write-Host "  YOUR SECURE LINK: " -ForegroundColor White -NoNewline
            Write-Host $matches[1] -ForegroundColor Magenta
            Write-Host "==================================================================
" -ForegroundColor Magenta
        } else {
            # Print other logs in dark gray so the purple stands out more!
            Write-Host $_ -ForegroundColor DarkGray
        }
    }
} finally {
    if (!$cf.HasExited) {
        Stop-Process -Id $cf.Id -Force
    }
}
