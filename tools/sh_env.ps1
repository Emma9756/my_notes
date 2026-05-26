$gitBin = "C:\Program Files\Git\bin"
$gitCmd = "C:\Program Files\Git\cmd"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$gitBin*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$gitBin;$gitCmd", "User")
}

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

where.exe sh