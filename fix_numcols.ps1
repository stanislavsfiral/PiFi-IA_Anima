$p = 'C:\PiFi-IA_Anima\sfiral_neurocore_torch.py'
Copy-Item $p "$p.bak_numcols" -Force
$L = Get-Content -Path $p -Encoding UTF8
$m1 = $L | Select-String -Pattern '^NUMERIC_COLS' -List
$m2 = $L | Select-String -Pattern '^def fit_meta' -List
if (-not $m1 -or -not $m2) { Write-Host 'MARKER NOT FOUND'; exit 1 }
$i = $m1.LineNumber - 1
$j = $m2.LineNumber - 1
$new = @(
'NUMERIC_COLS = [',
'    "n_nodes_log",',
'    "n_edges_log",',
'    "density",',
'    "max_degree_log",',
'    "coverage",',
'    "n_pairs_log",',
'    "pos_frac",',
'    "neg_frac",',
'    "zero_frac",',
'    "mean_sigma",',
'    "contrast",',
'    "antisym_frac",',
'    "sens_delta",',
'    "macro_R",',
'    "macro_L",',
'    "macro_S",',
'    "addr_entropy",',
'    "mean_scale_norm",',
']',
'',
'CAT_SPECS = {',
'    "tau": [',
'        "contact",',
'        "Axis X",',
'        "Axis Y",',
'        "Axis Z",',
'    ],',
'    "W_parity": ["-1", "0", "1"],',
'    "bipartite": ["0", "1"],',
'    "coherent": ["0", "1"],',
'    "fractal_ladder": ["0", "1"],',
'    "has_flip": ["0", "1"],',
'    "traversal_source": [',
'        "chain_seam",',
'        "nearest_lattice",',
'        "nearest_path",',
'        "empty",',
'    ],',
'}',
''
)
$out = @()
if ($i -gt 0) { $out += $L[0..($i-1)] }
$out += $new
$out += $L[$j..($L.Count-1)]
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($p, [string[]]$out, $enc)
Write-Host ('OK replaced lines ' + ($i+1) + ' to ' + $j)