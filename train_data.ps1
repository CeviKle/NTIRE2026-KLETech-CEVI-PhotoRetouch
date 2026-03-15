$BASE_YAML = "options/train/InRetouch_Optimize_Single2.yml"
$TEMP_YAML = "options/train/temp.yml"
$DATA_ROOT = "D:\saakshi\NTIRE2026-KLETech-CEVI-PhotoRetouch\Automatic_Evaluation_Data"  #chnage this path

# Get all sample directories starting with 'sample'
$sampleDirs = Get-ChildItem -Path $DATA_ROOT -Directory | Where-Object { $_.Name -like 'sample*' }

foreach ($sampleDir in $sampleDirs) {

    $sample = $sampleDir.Name

    $before = Join-Path $sampleDir.FullName "${sample}_before.jpg"
    $after  = Join-Path $sampleDir.FullName "${sample}_after.jpg"
    $input  = Join-Path $sampleDir.FullName "${sample}_input.jpg"

    $expName = "InRetouch_${sample}"

    Write-Host "Processing $sample"

    # Copy base YAML to temp YAML
    Copy-Item -Path $BASE_YAML -Destination $TEMP_YAML -Force

    # Replace lines in temp YAML:
    # Note: -replace is regex-based; use [regex]::Escape() for literal strings if needed

    # Read the temp YAML content
    $yamlContent = Get-Content $TEMP_YAML

    # Replace 'name: ...'
    $yamlContent = $yamlContent -replace 'name:.*', "name: $expName"

    # Replace style_natural line (adjust quotes for YAML array)
    $yamlContent = $yamlContent -replace 'style_natural:.*', "style_natural: ['${before}']"

    # Replace style_output line
    $yamlContent = $yamlContent -replace 'style_output:.*', "style_output: ['${after}']"

    # Replace inp_natural line
    $yamlContent = $yamlContent -replace 'inp_natural:.*', "inp_natural: '$input'"

    # Save changes back to temp YAML
    Set-Content -Path $TEMP_YAML -Value $yamlContent

    # Run the python training command with the temp YAML
    python -m basicsr.train_INR -opt $TEMP_YAML
}
