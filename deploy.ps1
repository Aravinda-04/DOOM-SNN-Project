param (
    [Parameter(Mandatory=$true, HelpMessage="Enter your username for the department server (e.g., e16xxx)")]
    [string]$Username,

    [Parameter(Mandatory=$true, HelpMessage="Enter the server address (e.g., tesla.ce.pdn.ac.lk)")]
    [string]$Server
)

$RemoteDir = "~/doom-snn-project"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " DOOM SNN: Automated Server Deployment" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Compress the necessary files locally to avoid uploading heavy cache folders (like .pytest_cache or venv)
Write-Host "`n[1/4] Packing files..." -ForegroundColor Yellow
# We only pack the essential source code and WAD file to keep the upload fast
tar.exe -czf deploy_doom.tar.gz src scenarios tests freedoom2.wad pytest.ini *.py 2>$null

# 2. Upload to the server via SCP (Will prompt for password)
Write-Host "`n[2/4] Uploading to $Server (You will be prompted for your password)..." -ForegroundColor Yellow
scp deploy_doom.tar.gz "${Username}@${Server}:~/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed. Please check your credentials." -ForegroundColor Red
    Remove-Item deploy_doom.tar.gz -ErrorAction SilentlyContinue
    exit
}

# 3. Clean up the local archive
Remove-Item deploy_doom.tar.gz -ErrorAction SilentlyContinue

# 4. SSH into the server, extract, install dependencies, and start tmux (Will prompt for password again)
Write-Host "`n[3/4] Connecting to server to set up the environment and start training..." -ForegroundColor Yellow
Write-Host "      (You will be prompted for your password one more time)" -ForegroundColor Yellow

$RemoteScript = @"
    echo '=> Extracting files...'
    mkdir -p $RemoteDir
    tar -xzf ~/deploy_doom.tar.gz -C $RemoteDir
    rm ~/deploy_doom.tar.gz

    cd $RemoteDir

    echo '=> Setting up Python Virtual Environment...'
    python3 -m venv venv
    source venv/bin/activate

    echo '=> Installing Dependencies (This might take a few minutes the first time)...'
    pip install --upgrade pip --quiet
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --quiet
    pip install snntorch vizdoom tensorboard numpy matplotlib pytest opencv-python-headless --quiet

    echo '=> Checking for existing tmux session...'
    tmux has-session -t doom_training 2>/dev/null
    
    if [ `$? != 0 ]; then
        echo '=> Starting training in a new tmux session background...'
        tmux new-session -d -s doom_training 'source venv/bin/activate; python src/train_level.py --model crsnn --stage full --run-id crsnn-map01-server --episodes 2000 --seed 0 --device cuda'
    else
        echo '=> tmux session "doom_training" already exists. Code was updated, but training was not restarted.'
    fi
"@

ssh "${Username}@${Server}" $RemoteScript

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host " Deployment Successful!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "The training is now running safely in the background on the server.`n"
Write-Host "To view the live output anytime, just run this command:"
Write-Host "    ssh ${Username}@${Server} -t `"tmux attach -t doom_training`"" -ForegroundColor Cyan
Write-Host "    (Remember to press Ctrl+B then D to disconnect without killing the training)`n"
