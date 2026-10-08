# Update only OTP provider settings from .env; preserve other service settings.
$ErrorActionPreference = 'Stop'
$otpEnvPath = Join-Path $PSScriptRoot '..\.env'
$otpValues = @{}
foreach ($otpLine in Get-Content -LiteralPath $otpEnvPath) {
    if ($otpLine -match '^\s*([A-Z_]+)\s*=(.*)$') {
        $otpValues[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}
$otpNames = @('ENGAGELO_API_KEY', 'WHATSAPP_PHONE_NUMBER_ID', 'ENGAGELO_OTP_TEMPLATE_ID')
foreach ($otpName in $otpNames) {
    if (-not $otpValues[$otpName]) { throw "Missing $otpName in .env" }
    if ($otpValues[$otpName].Contains('~')) { throw 'Unsupported delimiter in provider value' }
}
$otpServiceJson = & gcloud run services describe dilse-fastapi --project=aisteth-development --region=asia-south1 --format=json
if ($LASTEXITCODE -ne 0) { throw 'Google CLI authentication is required: gcloud auth login' }
$otpService = ($otpServiceJson -join "`n") | ConvertFrom-Json
$otpSecretNames = @($otpService.spec.template.spec.containers[0].env | Where-Object { $_.name -in $otpNames -and $_.valueFrom.secretKeyRef } | ForEach-Object { $_.name })
$otpUpdate = '^~^WHATSAPP_INTEGRATION_ENABLED=true~' + (($otpNames | ForEach-Object { $_ + '=' + $otpValues[$_] }) -join '~')
$otpArgs = @('run', 'services', 'update', 'dilse-fastapi', '--project=aisteth-development', '--region=asia-south1', '--update-env-vars', $otpUpdate, '--remove-env-vars', 'WHATSAPP_DEV_OTP,ALLOW_FIXED_OTP_IN_PRODUCTION', '--quiet')
if ($otpSecretNames.Count -gt 0) { $otpArgs += @('--remove-secrets', ($otpSecretNames -join ',')) }
& gcloud @otpArgs
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run OTP settings update failed' }
Write-Output 'Cloud Run OTP settings synchronized from .env. Credentials were not printed.'
