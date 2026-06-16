Add-Type -AssemblyName System.IO.Compression.FileSystem

$ErrorActionPreference = "Stop"

$Root = Resolve-Path "."
$SourcePpt = Get-ChildItem -Path $Root -Filter "*.pptx" |
    Where-Object { $_.Name -notlike "*_MediaPipe_added.pptx" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $SourcePpt) {
    throw "Source PPTX was not found."
}

$OutputName = "{0}_MediaPipe_added.pptx" -f $SourcePpt.BaseName
$OutputPpt = Join-Path $Root $OutputName
$WorkDir = Join-Path $Root "_tmp_mediapipe_pptx_build"
$VisualDir = Join-Path $Root "mediapipe_\ppt_visuals"

$RequiredImages = @(
    "mediapipe_capture_example_cropped.png",
    "mediapipe_frame_counts.png",
    "mediapipe_metrics_bar.png",
    "mediapipe_results_table.png",
    "mediapipe_confusion_matrices.png",
    "mediapipe_training_log_table.png"
)

foreach ($name in $RequiredImages) {
    $path = Join-Path $VisualDir $name
    if (-not (Test-Path $path)) {
        throw "Missing visual artifact: $path"
    }
}

if (Test-Path $WorkDir) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}
New-Item -ItemType Directory -Path $WorkDir | Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($SourcePpt.FullName, $WorkDir)

$PptDir = Join-Path $WorkDir "ppt"
$SlideDir = Join-Path $PptDir "slides"
$SlideRelDir = Join-Path $SlideDir "_rels"
$MediaDir = Join-Path $PptDir "media"
$PresentationPath = Join-Path $PptDir "presentation.xml"
$PresentationRelsPath = Join-Path $PptDir "_rels\presentation.xml.rels"
$ContentTypesPath = Join-Path $WorkDir "[Content_Types].xml"

function Escape-XmlText([string]$Value) {
    return [System.Security.SecurityElement]::Escape($Value)
}

function Text-Shape([string]$Id, [string]$Name, [int64]$X, [int64]$Y, [int64]$Cx, [int64]$Cy, [string]$Text, [int]$Size, [string]$Color, [bool]$Bold, [string]$Align = "l") {
    $b = if ($Bold) { '<a:b/>' } else { '' }
    $safe = Escape-XmlText $Text
    return @"
<p:sp>
  <p:nvSpPr><p:cNvPr id="$Id" name="$Name"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="$X" y="$Y"/><a:ext cx="$Cx" cy="$Cy"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" anchor="t"><a:spAutoFit/></a:bodyPr><a:lstStyle/>
    <a:p><a:pPr algn="$Align"/><a:r><a:rPr lang="ko-KR" sz="$($Size * 100)" dirty="0"><a:solidFill><a:srgbClr val="$Color"/></a:solidFill>$b</a:rPr><a:t>$safe</a:t></a:r><a:endParaRPr lang="ko-KR" sz="$($Size * 100)" dirty="0"/></a:p>
  </p:txBody>
</p:sp>
"@
}

function Rect-Shape([string]$Id, [string]$Name, [int64]$X, [int64]$Y, [int64]$Cx, [int64]$Cy, [string]$Fill, [string]$Line = "") {
    $lineXml = if ($Line) { "<a:ln><a:solidFill><a:srgbClr val=""$Line""/></a:solidFill></a:ln>" } else { "<a:ln><a:noFill/></a:ln>" }
    return @"
<p:sp>
  <p:nvSpPr><p:cNvPr id="$Id" name="$Name"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="$X" y="$Y"/><a:ext cx="$Cx" cy="$Cy"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="$Fill"/></a:solidFill>$lineXml</p:spPr>
  <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
</p:sp>
"@
}

function Picture-Shape([string]$Id, [string]$Name, [string]$Rid, [int64]$X, [int64]$Y, [int64]$Cx, [int64]$Cy) {
    return @"
<p:pic>
  <p:nvPicPr><p:cNvPr id="$Id" name="$Name"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="$Rid"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="$X" y="$Y"/><a:ext cx="$Cx" cy="$Cy"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>
"@
}

function Bullet-Shape([string]$Id, [string]$Name, [int64]$X, [int64]$Y, [int64]$Cx, [int64]$Cy, [string[]]$Items, [int]$Size = 15) {
    $paras = @()
    foreach ($item in $Items) {
        $safe = Escape-XmlText $item
        $sz = $Size * 100
        $paras += '<a:p><a:pPr marL="285750" indent="-171450"><a:buChar char="-"/></a:pPr><a:r><a:rPr lang="ko-KR" sz="{0}" dirty="0"><a:solidFill><a:srgbClr val="222222"/></a:solidFill></a:rPr><a:t>{1}</a:t></a:r><a:endParaRPr lang="ko-KR" sz="{0}" dirty="0"/></a:p>' -f $sz, $safe
    }
    $body = [string]::Join("`n", $paras)
    return @"
<p:sp>
  <p:nvSpPr><p:cNvPr id="$Id" name="$Name"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="$X" y="$Y"/><a:ext cx="$Cx" cy="$Cy"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square" anchor="t"><a:spAutoFit/></a:bodyPr><a:lstStyle/>$body</p:txBody>
</p:sp>
"@
}

function Make-SlideXml([string]$Title, [string]$Subtitle, [string[]]$BodyShapes) {
    $items = @()
    $items += Rect-Shape "2" "top accent" 0 0 1828800 91440 "BE1423"
    $items += Rect-Shape "3" "top line" 1828800 0 10363200 91440 "B0B0B0"
    $items += Text-Shape "4" "slide title" 777240 457200 10972800 548640 $Title 30 "141414" $true
    if ($Subtitle) {
        $items += Text-Shape "5" "slide subtitle" 792480 1036320 10668000 335280 $Subtitle 14 "5C5C5C" $false
    }
    $items += $BodyShapes
    $spTree = [string]::Join("`n", $items)
    return @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      $spTree
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"@
}

function Write-Rels([string]$Path, [string[]]$ImageTargets) {
    $rels = @()
    $rels += '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout7.xml"/>'
    for ($i = 0; $i -lt $ImageTargets.Count; $i++) {
        $rid = $i + 2
        $target = $ImageTargets[$i]
        $rels += '<Relationship Id="rId{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{1}"/>' -f $rid, $target
    }
    $body = [string]::Join("`n  ", $rels)
    $xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + "`n" + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "`n  " + $body + "`n</Relationships>"
    [IO.File]::WriteAllText($Path, $xml, [Text.UTF8Encoding]::new($false))
}

$ImageMap = @{
    "mediapipe_capture_example_cropped.png" = "image8.png"
    "mediapipe_frame_counts.png" = "image9.png"
    "mediapipe_metrics_bar.png" = "image10.png"
    "mediapipe_results_table.png" = "image11.png"
    "mediapipe_confusion_matrices.png" = "image12.png"
    "mediapipe_training_log_table.png" = "image13.png"
}

foreach ($entry in $ImageMap.GetEnumerator()) {
    Copy-Item -LiteralPath (Join-Path $VisualDir $entry.Key) -Destination (Join-Path $MediaDir $entry.Value) -Force
}

function Decode-B64([string]$Value) {
    return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Value))
}

$T = @{
    slide25Title = Decode-B64 "7KeB7KCRIOy0rOyYgSBNZWRpYVBpcGUg642w7J207YSwIOyYiOu5hCDsoIHsmqk="
    slide25Sub = Decode-B64 "7Iuk7KCcIOy0rOyYgSDtmZjqsr3sl5DshJwg6rSA7KCIIGxhbmRtYXJrIOy2lOy2nCDrsI8g67aE66WYIOqwgOuKpeyEseydhCDtmZXsnbjtlZwg7ZmV7J6lIOyLpO2XmA=="
    slide25Label = Decode-B64 "7KeB7KCRIOy0rOyYgSDsmIHsg4Hsl5DshJwgTWVkaWFQaXBlIGxhbmRtYXJr66W8IOy2lOy2nO2VtCDsi6Tsi5zqsIQg7IiY7KeRL+u2hOulmCDtmZTrqbTsnYQg6rWs7ISx7ZWY7JiA64ukLg=="
    slide25Purpose = Decode-B64 "7JiI67mEIOyggeyaqSDrqqnsoIE="
    s25b1 = Decode-B64 "QUktSHViIOq4sOuwmCDrs7jsi6Ttl5gg7J207ZuEIOyLpOygnCDstKzsmIEg7ZmY6rK9IOyggeyaqSDqsIDriqXshLHsnYQg7IKs7KCEIO2ZleyduA=="
    s25b2 = Decode-B64 "64yA7IOBIOyatOuPmTogc2lkZSBsdW5nZSwgYnVycGVlLCBwbGFuaywgcHVzaHVwLCBjcnVuY2g="
    s25b3 = Decode-B64 "7J6F66ClOiBNZWRpYVBpcGUgMzPqsJwgbGFuZG1hcmvsnZggeCwgeSwgeiDsooztkZw="
    s25b4 = Decode-B64 "7KCA7J6lIOuLqOychDog7JiB7IOBIOyDmO2UjOydtCDslYTri4jrnbwg7ZSE66CI7J6EIOuLqOychCBDU1Y="
    note = Decode-B64 "4oC7IOuzuCDqsrDqs7zripQg7ZSE66CI7J6EIOuLqOychCDsmIjruYQg7Iuk7ZeY7J2066mwLCBBSS1IdWIg67O47Iuk7ZeY7J2YIOyDmO2UjCDri6jsnIQg6rKw6rO87JmAIOyngeygkSDruYTqtZDtlZjsp4Ag7JWK64qU64ukLg=="
    slide26Title = Decode-B64 "TWVkaWFQaXBlIOqysOqzvCBDU1Yg7Iuc6rCB7ZmU"
    slide26Sub = Decode-B64 "7ZSE66CI7J6EIOyImCwg66qo6424IOyEseuKpSDsmpTslb0sIHRyYWluaW5nIGxvZ+ulvCBQUFQg7LKo67aA7JqpIOydtOuvuOyngOuhnCDsoJXrpqw="
    slide26Interp = Decode-B64 "Q1NWIOqysOqzvOuKlCDtlITroIjsnoQg7IiYLCDrqqjrjbjrs4Qg7JqU7JW9IOyEseuKpSwgdHJhaW5pbmcgbG9n66W8IOuwnO2RnOyaqSDqt7jrnpjtlIQv7ZGc66GcIOuzgO2ZmO2VmOyYgOuLpC4gWEdCb29zdOyZgCBTVk0g6rKw6rO864qUIOyYiOu5hCDsp4DtkZzroZwg7KCc7Iuc7ZWY65CYLCDrs7jsi6Ttl5jsnZggQUktSHViIOyDmO2UjCDri6jsnIQg7ISx64ql6rO8IOqwmeydgCDquLDspIDsnLzroZwg67mE6rWQ7ZWY7KeAIOyViuuKlOuLpC4="
    slide27Title = Decode-B64 "QUktSHViIOuzuOyLpO2XmOqzvOydmCDssKjsnbTsoJA="
    slide27Sub = Decode-B64 "642w7J207YSwIOq1rOyhsOyZgCDtj4nqsIAg64uo7JyE6rCAIOuLpOultOuvgOuhnCDsp4HsoJEg7ISx64qlIOu5hOq1kOqwgCDslYTri4jrnbwg7JiI67mEIOyggeyaqSDqsrDqs7zroZwg7ZW07ISd"
    diffTitle = Decode-B64 "QUktSHViIOuzuOyLpO2XmOqzvOydmCDssKjsnbQ="
    d1 = Decode-B64 "QUktSHViOiAyNOqwnCDqtIDsoIgg6riw67CYIC0zZC5qc29uLCBKU09OIDHqsJzrpbwg7Jq064+ZIOyImO2WiSDsg5jtlIzroZwg7IKs7Jqp"
    d2 = Decode-B64 "TWVkaWFQaXBlOiAzM+qwnCBsYW5kbWFyayDquLDrsJggQ1NWLCDqsIEg7ZaJ7J20IO2UhOugiOyehCDri6jsnIQg7J6F66Cl"
    d3 = Decode-B64 "QUktSHViOiAxN+qwnCDrp6jrqrjsmrTrj5kg652867Ko7J2YIOuzuOyLpO2XmCDqsrDqs7w="
    d4 = Decode-B64 "TWVkaWFQaXBlOiDsp4HsoJEg7LSs7JiBIDXqsJwg7Jq064+Z7J2YIO2ZleyepSDqsIDriqXshLEg7ZmV7J247JqpIOyYiOu5hCDsi6Ttl5g="
    confusionNote = Decode-B64 "7Zi864+Z7ZaJ66Cs7J2AIOyngeygkSDstKzsmIEg642w7J207YSw7JeQ7IScIOuqqOuNuCDsoIHsmqkg7Z2Q66aE7J2EIOuztOyXrOyjvOuKlCDrs7TsobAg7J6Q66OM7J2064ukLiDtmITsnqwg6rKw6rO8IO2MjOydvOyXkOyEnOuKlCBTVk3qs7wgWEdCb29zdOydmCDtj4nqsIAg7LSd65+J7J20IOyEnOuhnCDri6zrnbwg7KCV65+JIOu5hOq1kOuztOuLpCDsmIjruYQg67aE7ISdIOyekOujjOuhnCDtlbTshJ3tlZzri6Qu"
}

$slide25Bodies = @(
    Picture-Shape "10" "MediaPipe capture" "rId2" 762000 1645920 5791200 3657600
    Text-Shape "11" "capture label" 762000 5486400 5791200 396240 $T.slide25Label 14 "5C5C5C" $false "ctr"
    Rect-Shape "12" "summary box" 7010400 1645920 4429760 533400 "FFF6F7" "BE1423"
    Text-Shape "13" "summary title" 7240000 1767840 3962400 335280 $T.slide25Purpose 18 "BE1423" $true
    Bullet-Shape "14" "summary bullets" 7240000 2225040 3962400 1828800 @(
        $T.s25b1,
        $T.s25b2,
        $T.s25b3,
        $T.s25b4
    ) 15
    Text-Shape "15" "important note" 7010400 4876800 4429760 838200 $T.note 16 "BE1423" $true "ctr"
)

$slide26Bodies = @(
    Picture-Shape "10" "frame counts" "rId2" 731520 1584960 5105400 2537460
    Picture-Shape "11" "metrics" "rId3" 6553200 1584960 5105400 2537460
    Picture-Shape "12" "results table" "rId4" 1219200 4470000 4114800 1524000
    Picture-Shape "13" "training log table" "rId5" 6858000 4470000 4114800 1524000
    Text-Shape "14" "interpretation" 792480 6096000 10515600 487680 $T.slide26Interp 14 "5C5C5C" $false "ctr"
)

$slide27Bodies = @(
    Picture-Shape "10" "confusion matrices" "rId2" 853440 1463040 5715000 2926080
    Rect-Shape "11" "diff box" 6806160 1463040 4572000 2926080 "F7F7F7" "D9D9D9"
    Text-Shape "12" "diff title" 7040880 1615440 4114800 335280 $T.diffTitle 18 "141414" $true
    Bullet-Shape "13" "diff bullets" 7040880 2057400 4114800 1676400 @(
        $T.d1,
        $T.d2,
        $T.d3,
        $T.d4
    ) 14
    Text-Shape "14" "confusion note" 914400 4602480 5486400 853440 $T.confusionNote 14 "5C5C5C" $false "ctr"
    Text-Shape "15" "mandatory note" 6858000 4815840 4526280 914400 $T.note 17 "BE1423" $true "ctr"
)

[IO.File]::WriteAllText((Join-Path $SlideDir "slide25.xml"), (Make-SlideXml $T.slide25Title $T.slide25Sub $slide25Bodies), [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $SlideDir "slide26.xml"), (Make-SlideXml $T.slide26Title $T.slide26Sub $slide26Bodies), [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $SlideDir "slide27.xml"), (Make-SlideXml $T.slide27Title $T.slide27Sub $slide27Bodies), [Text.UTF8Encoding]::new($false))

Write-Rels (Join-Path $SlideRelDir "slide25.xml.rels") @("image8.png")
Write-Rels (Join-Path $SlideRelDir "slide26.xml.rels") @("image9.png", "image10.png", "image11.png", "image13.png")
Write-Rels (Join-Path $SlideRelDir "slide27.xml.rels") @("image12.png")

$contentTypes = [IO.File]::ReadAllText($ContentTypesPath)
foreach ($n in 25..27) {
    $override = '<Override PartName="/ppt/slides/slide{0}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' -f $n
    if ($contentTypes -notmatch [Regex]::Escape("/ppt/slides/slide$n.xml")) {
        $contentTypes = $contentTypes -replace "</Types>", "$override</Types>"
    }
}
[IO.File]::WriteAllText($ContentTypesPath, $contentTypes, [Text.UTF8Encoding]::new($false))

$presRels = [IO.File]::ReadAllText($PresentationRelsPath)
foreach ($n in 25..27) {
    $rid = "rId$($n + 100)"
    $rel = '<Relationship Id="{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{1}.xml"/>' -f $rid, $n
    if ($presRels -notmatch [Regex]::Escape(('Target="slides/slide{0}.xml"' -f $n))) {
        $presRels = $presRels -replace "</Relationships>", "$rel</Relationships>"
    }
}
[IO.File]::WriteAllText($PresentationRelsPath, $presRels, [Text.UTF8Encoding]::new($false))

$pres = [IO.File]::ReadAllText($PresentationPath)
$newSldIds = @(
    '<p:sldId id="1256" r:id="rId125"/>',
    '<p:sldId id="1257" r:id="rId126"/>',
    '<p:sldId id="1258" r:id="rId127"/>'
) -join ""
if ($pres -notmatch 'r:id="rId125"') {
    $pres = $pres -replace '(<p:sldId[^>]+r:id="rId23"\s*/>)', "`$1$newSldIds"
}
[IO.File]::WriteAllText($PresentationPath, $pres, [Text.UTF8Encoding]::new($false))

if (Test-Path $OutputPpt) {
    Remove-Item -LiteralPath $OutputPpt -Force
}
$outStream = [IO.File]::Open($OutputPpt, [IO.FileMode]::CreateNew)
$outZip = New-Object System.IO.Compression.ZipArchive($outStream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    Get-ChildItem -LiteralPath $WorkDir -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($WorkDir.Length + 1).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($outZip, $_.FullName, $relative) | Out-Null
    }
}
finally {
    $outZip.Dispose()
    $outStream.Dispose()
}

$zip = [System.IO.Compression.ZipFile]::OpenRead($OutputPpt)
$slideCount = ($zip.Entries | Where-Object { $_.FullName -match '^ppt[\\/]slides[\\/]slide[0-9]+\.xml$' }).Count
$entryNames = @($zip.Entries | ForEach-Object { $_.FullName })
$hasSlide25 = $entryNames -contains "ppt/slides/slide25.xml" -or $entryNames -contains "ppt\slides\slide25.xml"
$hasSlide26 = $entryNames -contains "ppt/slides/slide26.xml" -or $entryNames -contains "ppt\slides\slide26.xml"
$hasSlide27 = $entryNames -contains "ppt/slides/slide27.xml" -or $entryNames -contains "ppt\slides\slide27.xml"
$zip.Dispose()

if ($slideCount -ne 27 -or -not $hasSlide25 -or -not $hasSlide26 -or -not $hasSlide27) {
    throw "PPTX validation failed. slideCount=$slideCount"
}

Remove-Item -LiteralPath $WorkDir -Recurse -Force
Write-Host "Created: $OutputPpt"
Write-Host "Slide count: $slideCount"
