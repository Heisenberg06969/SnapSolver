@echo off
echo Starting SnapSolver Backend on port 8080...
start cmd /k "python backend.py"

echo ========================================================
echo Generating a secure HTTPS link via Cloudflare...
echo Look for the URL that ends in ".trycloudflare.com" and open it on your phone!
echo ========================================================
start pwsh -NoProfile -ExecutionPolicy Bypass -Command ".\start_cloudflare.ps1"
